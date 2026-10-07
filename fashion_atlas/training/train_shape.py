import argparse
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fashion_atlas.common import DATA, CACHE, ARTIFACTS, read_json, record_metrics
from fashion_atlas.models import ShapeAutoencoder
from fashion_atlas.training.utils import setup, metadata_retrieval, supervised_contrastive


@torch.inference_mode()
def reconstruct(model, images, indices):
    model.eval()
    errors, overlaps = [], []
    for batch in indices.split(64):
        target = images[batch].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(target)
        errors.append((logits.float().sigmoid() - target).square().mean((1, 2, 3)))
        prediction = logits > 0
        overlaps.append(
            (prediction & target.bool()).sum((1, 2, 3))
            / (prediction | target.bool()).sum((1, 2, 3)).clamp_min(1)
        )
    return {
        "mse": torch.cat(errors).mean().item(),
        "iou": torch.cat(overlaps).mean().item(),
    }


def main(epochs=150):
    device = setup()
    torch.backends.cudnn.benchmark = True
    start = time.perf_counter()
    items = read_json(DATA / "items.json")
    cache = np.load(CACHE / "images.npz")
    split = np.array([x["split"] for x in items])
    valid = cache["valid"][:, 0]
    train = np.flatnonzero(valid & (split == "train"))
    evaluation = np.flatnonzero(
        valid & (split == "evaluation") & cache["evaluation_safe"]
    )
    images = torch.as_tensor(cache["masks"][:, None], device=device)
    train_ids = torch.as_tensor(train, device=device)
    eval_ids = torch.as_tensor(evaluation, device=device)
    model = ShapeAutoencoder().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00005
    )
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_ids[torch.randperm(len(train_ids), device=device)].split(64):
            target = images[batch].float().contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(target)
            loss = F.mse_loss(logits.float().sigmoid(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        metrics = reconstruct(model, images, eval_ids)
        history.append({"epoch": epoch, **metrics})
        if metrics["mse"] < best:
            meaningful = metrics["mse"] < best - 0.00001
            best = metrics["mse"]
            stale = 0 if meaningful else stale + 1
            torch.save(model.state_dict(), ARTIFACTS / "shape_autoencoder.pt")
        else:
            stale += 1
        if epoch % 10 == 0:
            print("Shape", epoch, metrics, flush=True)
        if (epoch >= 50 and stale >= 30) or time.perf_counter() - start > 720:
            break
    model.load_state_dict(
        torch.load(ARTIFACTS / "shape_autoencoder.pt", weights_only=True)
    )
    raw = np.zeros((len(items), 64), np.float32)
    with torch.inference_mode():
        for batch in torch.as_tensor(np.flatnonzero(valid), device=device).split(128):
            raw[batch.cpu().numpy()] = (
                F.normalize(model.encode(images[batch].float()), dim=1).cpu().numpy()
            )
    reconstruction = reconstruct(model, images, eval_ids)
    baseline = metadata_retrieval(raw, items, evaluation, train)
    features = torch.as_tensor(raw, device=device)
    labels = []
    for field in ("category", "subcategory"):
        values = [
            x[field] if field == "category" else x["category"] + "|" + x[field]
            for x in items
        ]
        vocabulary = {v: i for i, v in enumerate(sorted({values[j] for j in train}))}
        labels.append(
            torch.tensor([vocabulary.get(v, -1) for v in values], device=device)
        )
    projection = nn.Linear(64, 64, bias=False).to(device)
    projection.weight.data.copy_(torch.eye(64, device=device))
    optimizer = torch.optim.AdamW(projection.parameters(), lr=0.001, weight_decay=0.001)
    best, stale, projection_history = float(np.mean(list(baseline.values()))), 0, []
    torch.save(projection.state_dict(), ARTIFACTS / "shape_projection.pt")
    for epoch in range(1, 151):
        for batch in train_ids[torch.randperm(len(train_ids), device=device)].split(
            256
        ):
            x = features[batch]
            z = F.normalize(projection(x), dim=1)
            loss = 0.35 * supervised_contrastive(
                z, labels[0][batch]
            ) + 0.65 * supervised_contrastive(z, labels[1][batch])
            loss += (
                2 * (1 - (z * x).sum(1)).mean()
                + 0.5 * (z @ z.T - x @ x.T).square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            with torch.inference_mode():
                embeddings = F.normalize(projection(features), dim=1).cpu().numpy()
            metrics = metadata_retrieval(embeddings, items, evaluation, train)
            score = np.mean(list(metrics.values()))
            projection_history.append({"epoch": epoch, **metrics})
            if score > best + 1e-6:
                best, stale = score, 0
                torch.save(projection.state_dict(), ARTIFACTS / "shape_projection.pt")
            else:
                stale += 5
            if stale >= 30:
                break
    projection.load_state_dict(
        torch.load(ARTIFACTS / "shape_projection.pt", weights_only=True)
    )
    with torch.inference_mode():
        embeddings = F.normalize(projection(features), dim=1).cpu().numpy()
    np.savez_compressed(CACHE / "shape_embeddings.npz", embeddings=embeddings)
    record_metrics(
        "shape",
        {
            "seconds": time.perf_counter() - start,
            "train_images": len(train),
            "validation_images": len(evaluation),
            "reconstruction": reconstruction,
            "baseline_p5": baseline,
            "projected_p5": metadata_retrieval(embeddings, items, evaluation, train),
            "history": history,
            "projection_history": projection_history,
        },
    )
    print("Shape training complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    main(**vars(parser.parse_args()))
