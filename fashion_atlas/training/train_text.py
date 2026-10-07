import argparse
import itertools
import json
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from fashion_atlas.common import DATA, CACHE, ARTIFACTS, read_json, record_metrics
from fashion_atlas.models import TextHeads, load_text_backbone, text_features
from fashion_atlas.training.utils import setup


def training_rows(items):
    by_name = {item["name"]: item for item in items}
    safe = np.load(CACHE / "images.npz")["evaluation_safe"]
    rows, skipped = [], Counter()
    attributes = ("palette", "pattern", "shape")
    with (DATA / "queries.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            item = by_name.get(record["name"])
            if item is None:
                continue
            if item["split"] != "train" and not safe[item["source_id"]]:
                skipped["duplicate_validation_image"] += len(record["queries"])
                continue
            for query in record["queries"]:
                mask = [float(bool(query["attribute_mask"][key])) for key in attributes]
                reason = None
                if not any(mask):
                    reason = "no_active_attribute"
                elif query.get("warnings"):
                    reason = "query_warning"
                elif any(
                    mask[j] and not (query.get("evidence") or {}).get(key)
                    for j, key in enumerate(attributes)
                ):
                    reason = "missing_evidence"
                if reason:
                    skipped[reason] += 1
                    continue
                weights = np.array(
                    [query["weights"][key] for key in attributes], np.float32
                )
                if (
                    not np.isfinite(weights).all()
                    or weights.min() < 0
                    or weights.sum() <= 0
                ):
                    skipped["invalid_weights"] += 1
                    continue
                rows.append(
                    {
                        "text": query["text"],
                        "id": item["id"],
                        "name": item["name"],
                        "style": item["style"],
                        "split": item["split"],
                        "mask": mask,
                        "weights": (weights / weights.sum()).tolist(),
                        "kind": query["kind"],
                    }
                )
    return rows, dict(skipped)


def attribute_losses(prediction, target, permutations):
    colors = prediction[:, :16].reshape(-1, 4, 4)[:, None]
    expected = target[:, :16].reshape(-1, 4, 4)[:, permutations]
    rgb_error = (colors[:, :, :, :3] - expected[:, :, :, :3]).square().mean(-1)
    mass_error = (colors[:, :, :, 3] - expected[:, :, :, 3]).square().mean(-1)
    palette = ((rgb_error * expected[:, :, :, 3]).sum(-1) + mass_error).min(1).values
    pattern = 1 - (prediction[:, 16:80] * target[:, 16:80]).sum(1)
    shape = 1 - (prediction[:, 80:] * target[:, 80:]).sum(1)
    return torch.stack([palette, pattern, shape], dim=1)


def main(frozen_epochs=40, tuned_epochs=8):
    device = setup()
    start = time.perf_counter()
    items = read_json(ARTIFACTS / "catalog.json")["items"]
    rows, excluded = training_rows(items)
    train = torch.tensor(
        [i for i, r in enumerate(rows) if r["split"] == "train"], device=device
    )
    validation = torch.tensor(
        [i for i, r in enumerate(rows) if r["split"] == "evaluation"], device=device
    )
    assert set(rows[i]["style"] for i in train.tolist()).isdisjoint(
        rows[i]["style"] for i in validation.tolist()
    )
    target = torch.tensor(
        np.array([items[r["id"]]["descriptor"] for r in rows]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.tensor([r["mask"] for r in rows], device=device)
    weights = torch.tensor([r["weights"] for r in rows], device=device)
    emphasis = weights / weights.max(1, keepdim=True).values.clamp_min(1e-8)
    permutations = torch.tensor(list(itertools.permutations(range(4))), device=device)
    tokenizer, backbone = load_text_backbone(device)
    texts = [row["text"] for row in rows]
    encoded = []
    for offset in range(0, len(texts), 256):
        encoded.append(
            text_features(texts[offset : offset + 256], tokenizer, backbone).cpu()
        )
        if offset % 10240 == 0:
            print(f"Text features {offset}/{len(texts)}", flush=True)
    features = torch.cat(encoded).to(device)
    heads = TextHeads().to(device)
    tokens = None

    def forward(indices, tuned):
        if tuned:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = backbone(
                    **{key: value[indices] for key, value in tokens.items()}
                ).text_embeds
                predicted, logits = heads(F.normalize(output.float(), dim=1))
            return predicted.float(), logits.float()
        return heads(features[indices])

    @torch.inference_mode()
    def evaluate(tuned):
        heads.eval()
        backbone.eval()
        loss_sum = torch.zeros(3, device=device)
        weight_error = torch.zeros(3, device=device)
        for indices in validation.split(256):
            predicted, logits = forward(indices, tuned)
            loss_sum += (
                attribute_losses(predicted, target[indices], permutations)
                * mask[indices]
            ).sum(0)
            probability = logits.sigmoid()
            probability /= probability.sum(1, keepdim=True)
            weight_error += (probability - weights[indices]).abs().sum(0)
        return {
            "losses": (loss_sum / mask[validation].sum(0).clamp_min(1)).tolist(),
            "weight_mae": (weight_error / len(validation)).tolist(),
        }

    stages = {}
    for tuned, epochs in ((False, frozen_epochs), (True, tuned_epochs)):
        stage_start = time.perf_counter()
        name = "tuned" if tuned else "frozen"
        if tuned:
            heads.load_state_dict(
                torch.load(ARTIFACTS / "text_heads_frozen.pt", weights_only=True)
            )
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            for key, parameter in backbone.named_parameters():
                if key.startswith(
                    (
                        "text_model.encoder.layers.10.",
                        "text_model.encoder.layers.11.",
                        "text_model.final_layer_norm.",
                        "text_projection.",
                    )
                ):
                    parameter.requires_grad_(True)
            parameters = [p for p in backbone.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": parameters, "lr": 5e-6},
                    {"params": heads.parameters(), "lr": 1e-4},
                ],
                weight_decay=0.01,
            )
        else:
            parameters = []
            optimizer = torch.optim.AdamW(
                heads.parameters(), lr=0.001, weight_decay=0.01
            )
        checkpoint = ARTIFACTS / ("text_heads.pt" if tuned else "text_heads_frozen.pt")

        def save():
            torch.save(heads.state_dict(), checkpoint)
            if tuned:
                delta = {
                    key: parameter.detach().cpu()
                    for key, parameter in backbone.named_parameters()
                    if parameter.requires_grad
                }
                torch.save(delta, ARTIFACTS / "text_delta.pt")

        initial = evaluate(tuned)
        best, best_epoch, history = sum(initial["losses"]), 0, []
        save()
        for epoch in range(1, epochs + 1):
            heads.train()
            if tuned:
                backbone.train()
            order = train[torch.randperm(len(train), device=device)]
            for indices in order.split(128 if tuned else 512):
                predicted, logits = forward(indices, tuned)
                loss = (
                    attribute_losses(predicted, target[indices], permutations)
                    * mask[indices]
                ).sum() / mask[indices].sum().clamp_min(1)
                loss += 0.3 * F.binary_cross_entropy_with_logits(
                    logits, emphasis[indices]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters + list(heads.parameters()), 1)
                optimizer.step()
            result = evaluate(tuned)
            history.append({"epoch": epoch, **result})
            score = sum(result["losses"])
            if score < best:
                best, best_epoch = score, epoch
                save()
            print("Text", name, epoch, result, flush=True)
            if epoch - best_epoch >= (3 if tuned else 7):
                break
        stages[name] = {
            "seconds": time.perf_counter() - stage_start,
            "initial": initial,
            "best_epoch": best_epoch,
            "best": history[best_epoch - 1] if best_epoch else initial,
            "history": history,
            "backbone_trainable_parameters": sum(p.numel() for p in parameters),
        }
    record_metrics(
        "text",
        {
            "seconds": time.perf_counter() - start,
            "queries": len(rows),
            "train_queries": len(train),
            "validation_queries": len(validation),
            "excluded": excluded,
            **stages,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-epochs", type=int, default=40)
    parser.add_argument("--tuned-epochs", type=int, default=8)
    main(**vars(parser.parse_args()))
