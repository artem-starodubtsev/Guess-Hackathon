"""RGB garment image -> normalized binary silhouette; no file IO or neural nets."""

from __future__ import annotations
import cv2
import numpy as np
from fashion_atlas.preprocessing.segmentation import _garment_mask

__all__ = ["extract_shape"]


def extract_shape(
    image: np.ndarray,
    output_size: tuple[int, int] = (256, 256),
    *,
    padding: float = 0.05,
) -> np.ndarray:
    """Return a uint8 (height, width) array: garment=1, background=0."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy RGB array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise ValueError("image must have shape (H, W, 3), with H and W >=16")
    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise TypeError("output_size must be a (width, height) tuple")
    if any(
        (
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in output_size
        )
    ):
        raise TypeError("output_size values must be integers")
    width, height = map(int, output_size)
    if min(width, height) <= 0:
        raise ValueError("output_size must be positive")
    if isinstance(padding, (bool, np.bool_)) or not isinstance(
        padding, (int, float, np.number)
    ):
        raise TypeError("padding must be a real number")
    if not np.isfinite(padding) or not 0 <= padding <= 0.4:
        raise ValueError("padding must be between 0 and 0.4")
    h, w = image.shape[:2]
    scale = min(1.0, 512 / max(h, w))
    rgb = np.ascontiguousarray(image)
    if scale < 1:
        rgb = cv2.resize(
            rgb,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    mask = _garment_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    mask = _clean_mask(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No foreground remains after segmentation")
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    inner_w = max(1, width - 2 * round(width * padding))
    inner_h = max(1, height - 2 * round(height * padding))
    factor = min(inner_w / crop.shape[1], inner_h / crop.shape[0])
    rw = min(inner_w, max(1, round(crop.shape[1] * factor)))
    rh = min(inner_h, max(1, round(crop.shape[0] * factor)))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((height, width), np.uint8)
    x, y = ((width - rw) // 2, (height - rh) // 2)
    result[y : y + rh, x : x + rw] = resized > 0
    return result


def _clean_mask(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if n <= 1:
        raise ValueError("No garment found")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    n, holes, stats, _ = cv2.connectedComponentsWithStats(1 - mask, 8)
    edge_labels = np.unique(
        np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
    )
    max_hole = max(1, round(int(mask.sum()) * 0.001))
    fill = np.zeros(n, dtype=bool)
    fill[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_hole
    fill[edge_labels] = False
    mask[fill[holes]] = 1
    return mask
