import random
import numpy as np
import torch
from torch.nn import functional as F


def setup(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    return "cuda"


def unit(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def supervised_contrastive(embeddings, labels):
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~eye & (labels[:, None] >= 0)
    valid = positives.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    logits = (embeddings @ embeddings.T / 0.12).masked_fill(eye, -1e4)
    loss = -(F.log_softmax(logits, dim=1) * positives).sum(1) / positives.sum(
        1
    ).clamp_min(1)
    return loss[valid].mean()


def metadata_retrieval(embeddings, items, queries, gallery):
    similarities = unit(embeddings[queries]) @ unit(embeddings[gallery]).T
    styles = np.array([x["style"] for x in items])
    similarities[styles[queries, None] == styles[gallery][None, :]] = -10
    neighbors = np.asarray(gallery)[
        np.argsort(-similarities, axis=1, kind="stable")[:, :5]
    ]
    metrics = {}
    for field in ("category", "subcategory", "construction"):
        values = np.array(
            [
                x[field] if field == "category" else x["category"] + "|" + x[field]
                for x in items
            ]
        )
        eligible = np.any(
            (values[queries, None] == values[gallery][None, :])
            & (styles[queries, None] != styles[gallery][None, :]),
            axis=1,
        )
        eligible &= np.array([items[i][field] not in ("", "-") for i in queries])
        match = values[queries, None] == values[neighbors]
        metrics[field] = float(match[eligible].mean()) if eligible.any() else 0.0
    return metrics
