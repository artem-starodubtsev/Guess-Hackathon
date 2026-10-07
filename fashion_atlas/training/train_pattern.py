import time
import numpy as np
import torch
from torch.nn import functional as F

from fashion_atlas.common import DATA, CACHE, ARTIFACTS, read_json, record_metrics, product_key
from fashion_atlas.models import PatternProjection, load_dino, dino_features
from fashion_atlas.training.utils import setup, unit


def crop_retrieval(features, products):
    scores = features[:, 0] @ features[:, 2].T
    order = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    hits = products[order] == products[:, None]
    return {"r1": float(hits[:, 0].mean()), "r5": float(hits.any(1).mean())}


def main():
    device = setup()
    start = time.perf_counter()
    items = read_json(DATA / "items.json")
    cache = np.load(CACHE / "images.npz")
    valid_ids = np.flatnonzero(cache["valid"][:, 1])
    feature_path = CACHE / "pattern_features.npz"
    if feature_path.exists():
        features = np.load(feature_path)["features"]
    else:
        dino = load_dino(device)
        features = np.zeros((len(items), 3, 384), np.float32)
        crops = cache["crops"]
        for offset in range(0, len(valid_ids), 128):
            ids = valid_ids[offset : offset + 128]
            features[ids] = dino_features(
                dino, crops[ids].reshape(-1, 128, 128)
            ).reshape(-1, 3, 384)
            print(
                f"DINO features {min(offset+128,len(valid_ids))}/{len(valid_ids)}",
                flush=True,
            )
        np.savez_compressed(feature_path, features=features)
        del dino
        torch.cuda.empty_cache()
    train = np.array([i for i in valid_ids if items[i]["split"] == "train"])
    evaluation = np.array(
        [
            i
            for i in valid_ids
            if items[i]["split"] == "evaluation" and cache["evaluation_safe"][i]
        ]
    )
    flat = features[train].reshape(-1, 384)
    mean = flat.mean(0)
    _, _, components = np.linalg.svd(flat - mean, full_matrices=False)
    model = PatternProjection(mean, components[:64]).to(device)
    x = torch.tensor(features, device=device)
    with torch.no_grad():
        teacher = model(x).detach().clone()
    products = [product_key(item) for item in items]
    vocabulary = {value: i for i, value in enumerate(sorted(set(products)))}
    product_ids = np.array([vocabulary[p] for p in products])
    labels = torch.tensor(product_ids, device=device)
    colors = {
        value: i for i, value in enumerate(sorted({item["color"] for item in items}))
    }
    color_labels = torch.tensor(
        [colors[item["color"]] for item in items], device=device
    )
    baseline = crop_retrieval(
        teacher[evaluation].cpu().numpy(), product_ids[evaluation]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 150, eta_min=0.00003
    )
    train_ids = torch.tensor(train, device=device)
    best = baseline["r1"] + 0.25 * baseline["r5"]
    torch.save(model.state_dict(), ARTIFACTS / "pattern_projection.pt")
    stale, history = 0, []
    for epoch in range(1, 151):
        for batch in train_ids[torch.randperm(len(train_ids), device=device)].split(
            128
        ):
            z = model(x[batch]).reshape(-1, 64)
            product = labels[batch].repeat_interleave(3)
            color = color_labels[batch].repeat_interleave(3)
            diagonal = torch.eye(len(z), device=device, dtype=torch.bool)
            positive = (product[:, None] == product[None, :]) & ~diagonal
            ambiguous = (color[:, None] == color[None, :]) & (
                product[:, None] != product[None, :]
            )
            logits = (z @ z.T / 0.1).masked_fill(diagonal | ambiguous, -1e4)
            log_prob = F.log_softmax(logits, dim=1)
            loss = (
                -(log_prob * positive).sum(1).div(positive.sum(1).clamp_min(1)).mean()
            )
            loss += 0.5 * (1 - (z * teacher[batch].reshape(-1, 64)).sum(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0:
            with torch.inference_mode():
                projected = model(x[evaluation]).cpu().numpy()
            metrics = crop_retrieval(projected, product_ids[evaluation])
            history.append({"epoch": epoch, **metrics})
            score = metrics["r1"] + 0.25 * metrics["r5"]
            if score > best + 1e-6:
                best, stale = score, 0
                torch.save(model.state_dict(), ARTIFACTS / "pattern_projection.pt")
            else:
                stale += 5
            print("Pattern", epoch, metrics, flush=True)
            if stale >= 30:
                break
    model.load_state_dict(
        torch.load(ARTIFACTS / "pattern_projection.pt", weights_only=True)
    )
    with torch.inference_mode():
        projected = model(x).cpu().numpy()
    embeddings = unit(projected.mean(1))
    embeddings[~cache["valid"][:, 1]] = 0
    np.savez_compressed(CACHE / "pattern_embeddings.npz", embeddings=embeddings)
    record_metrics(
        "pattern",
        {
            "seconds": time.perf_counter() - start,
            "train_images": len(train),
            "validation_images": len(evaluation),
            "pca_cross_crop": baseline,
            "learned_cross_crop": crop_retrieval(
                projected[evaluation], product_ids[evaluation]
            ),
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
