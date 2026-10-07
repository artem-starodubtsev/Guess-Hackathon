import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ARTIFACTS
from device import select_device
from models import (
    ShapeAutoencoder,
    PatternProjection,
    TextHeads,
    load_dino,
    dino_features,
    load_text_backbone,
    text_features,
)
from segmentation import _garment_mask
from shape import extract_shape
from pattern import extract_pattern, three_crops
from palette import extract_palette


class ImageEncoder:
    """RGB uint8 photo -> palette16, pattern64, shape64."""

    def __init__(self, device=None):
        cv2.setNumThreads(1)
        self.device = select_device(device)
        self.shape = ShapeAutoencoder().to(self.device).eval()
        self.shape.load_state_dict(
            torch.load(
                ARTIFACTS / "shape_autoencoder.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.shape_projection = nn.Linear(64, 64, bias=False).to(self.device).eval()
        self.shape_projection.load_state_dict(
            torch.load(
                ARTIFACTS / "shape_projection.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.dino = load_dino(self.device)
        self.pattern_projection = PatternProjection().to(self.device).eval()
        self.pattern_projection.load_state_dict(
            torch.load(
                ARTIFACTS / "pattern_projection.pt",
                map_location=self.device,
                weights_only=True,
            )
        )

    @torch.inference_mode()
    def encode(self, rgb, *, with_previews=False):
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError("Expected an RGB uint8 array with shape (height,width,3)")
        mask = cv2.resize(
            extract_shape(rgb), (128, 128), interpolation=cv2.INTER_NEAREST
        )
        crops, _, _ = three_crops(
            rgb, _garment_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        )
        palette = extract_palette(extract_pattern(rgb))["vector"]
        image = torch.tensor(mask[None, None], dtype=torch.float32, device=self.device)
        shape = F.normalize(self.shape.encode(image), dim=1)
        shape = F.normalize(self.shape_projection(shape), dim=1)[0]
        features = torch.tensor(dino_features(self.dino, crops), device=self.device)
        pattern = F.normalize(self.pattern_projection(features).mean(0), dim=0)
        descriptor = np.concatenate(
            [palette, pattern.cpu().numpy(), shape.cpu().numpy()]
        )
        if with_previews:
            return descriptor, mask, crops
        return descriptor


class TextEncoder:
    def __init__(self, tuned=True, device=None):
        self.device = select_device(device)
        self.tokenizer, self.backbone = load_text_backbone(self.device)
        if tuned:
            delta = torch.load(
                ARTIFACTS / "text_delta.pt", map_location=self.device, weights_only=True
            )
            self.backbone.load_state_dict(delta, strict=False)
        self.heads = TextHeads().to(self.device).eval()
        filename = "text_heads.pt" if tuned else "text_heads_frozen.pt"
        self.heads.load_state_dict(
            torch.load(
                ARTIFACTS / filename, map_location=self.device, weights_only=True
            )
        )

    @torch.inference_mode()
    def encode(self, texts):
        descriptor, logits = self.heads(
            text_features(texts, self.tokenizer, self.backbone)
        )
        raw = logits.sigmoid()
        weights = torch.where(raw < 0.1, 0, raw)
        weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, raw)
        weights /= weights.sum(1, keepdim=True)
        return descriptor.cpu().numpy(), weights.cpu().numpy()


def rank(items, distances, weights, exclude_product=None, group=True):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Provide three finite, nonnegative weights")
    if weights.sum() == 0:
        return []
    scores = distances @ weights / weights.sum()
    order = np.lexsort((np.arange(len(items)), np.floor(scores / 1e-6 + 0.5)))
    seen, results = set(), []
    for index in order:
        item = items[index]
        product = (item["style"], item["color"])
        if product == exclude_product or (group and product in seen):
            continue
        seen.add(product)
        results.append({"id": int(index), "distance": float(scores[index])})
    return results
