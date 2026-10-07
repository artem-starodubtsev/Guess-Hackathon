"""Automatic print extraction using only OpenCV and NumPy."""

from __future__ import annotations
import cv2
import numpy as np
from segmentation import _garment_mask

__all__ = ["extract_pattern"]


def extract_pattern(
    image: np.ndarray,
    output_size: tuple[int, int] = (256, 256),
    *,
    min_coverage: float = 0.9,
) -> np.ndarray:
    """Extract a filled print image from one garment on a white background."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array in RGB format")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8 with values in 0..255")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3), RGB")
    if min(image.shape[:2]) < 16:
        raise ValueError("image width and height must each be at least 16 pixels")
    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise TypeError("output_size must be a (width, height) tuple")
    if any(
        (
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in output_size
        )
    ):
        raise TypeError("output_size values must be integers")
    width, height = (int(v) for v in output_size)
    if width <= 0 or height <= 0:
        raise ValueError("output_size values must be positive")
    source = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    mask = _garment_mask(source)
    box, _ = _find_coverage_square(mask, target=min_coverage)
    canvas, known = _prepare_crop(source, mask, box)
    filled = _fill_texture(canvas, known)
    interpolation = cv2.INTER_AREA if max(width, height) <= 384 else cv2.INTER_CUBIC
    resized = cv2.resize(filled, (width, height), interpolation=interpolation)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _find_coverage_square(mask, target=0.9, search_side=256):
    """Largest sampled square with native-resolution foreground coverage > target."""
    if isinstance(target, (bool, np.bool_)) or not isinstance(
        target, (int, float, np.number)
    ):
        raise TypeError("target must be a real coverage fraction")
    if not np.isfinite(target) or not 0 < target < 1:
        raise ValueError("target must be strictly between 0 and 1")
    h, w = mask.shape
    scale = min(1.0, search_side / max(h, w))
    sh, sw = (max(1, round(h * scale)), max(1, round(w * scale)))
    binary = (mask > 0).astype(np.uint8)
    reduced = cv2.resize(
        binary.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
    )
    integral = cv2.integral(reduced, sdepth=cv2.CV_64F)
    original_integral = cv2.integral(binary, sdepth=cv2.CV_64F)
    moments = cv2.moments(reduced)
    if moments["m00"] == 0:
        raise ValueError("Empty garment mask")
    cx, cy = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    minimum_side = max(8, round(min(sh, sw) * 0.06))
    for size in range(min(sh, sw), minimum_side - 1, -2):
        sums = (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        ys, xs = np.where(sums / (size * size) > target)
        if not len(xs):
            continue
        side = min(round(size / scale), w, h)
        ox = np.clip(np.rint(xs / scale).astype(np.int64), 0, w - side)
        oy = np.clip(np.rint(ys / scale).astype(np.int64), 0, h - side)
        coverage = (
            original_integral[oy + side, ox + side]
            - original_integral[oy, ox + side]
            - original_integral[oy + side, ox]
            + original_integral[oy, ox]
        ) / (side * side)
        valid = coverage > target
        if not valid.any():
            continue
        best = np.max(coverage[valid])
        choices = np.flatnonzero(valid & (coverage >= best - 1e-12))
        choice = choices[
            np.argmin(
                (xs[choices] + size / 2 - cx) ** 2 + (ys[choices] + size / 2 - cy) ** 2
            )
        ]
        return ((int(ox[choice]), int(oy[choice]), side), float(coverage[choice]))
    raise ValueError("No sampled square exceeds requested garment coverage")


def _prepare_crop(image, mask, box, working_size=384):
    x, y, s = box
    raw = image[y : y + s, x : x + s]
    cropped_mask = mask[y : y + s, x : x + s]
    canvas = cv2.resize(
        raw,
        (working_size, working_size),
        interpolation=cv2.INTER_AREA if s >= working_size else cv2.INTER_CUBIC,
    )
    m = cv2.resize(
        cropped_mask, (working_size, working_size), interpolation=cv2.INTER_NEAREST
    )
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    canvas[m == 0] = 255
    return (canvas, m)


def _fill_texture(canvas, mask, source=None, source_mask=None, seed=17):
    """Fill only missing pixels; match boundaries against fully observed donors."""
    result = canvas.copy()
    known = mask > 0
    initial = known.copy()
    if source is None:
        source, source_mask = (canvas.copy(), mask.copy())
    src = source.astype(np.float32) / 255
    srcmask = (source_mask > 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    patch_size = 49
    while patch_size >= 13:
        coverage = cv2.matchTemplate(
            srcmask, np.ones((patch_size, patch_size), np.float32), cv2.TM_CCORR
        )
        valid = coverage > patch_size * patch_size - 0.1
        if np.count_nonzero(valid) >= 20:
            break
        patch_size -= 6
    if patch_size < 13:
        raise ValueError("Insufficient fully visible donor texture")
    uses = np.zeros(valid.shape, np.float32)
    half = patch_size // 2
    iterations = 0
    h, w = known.shape
    while not np.all(known):
        counts = cv2.boxFilter(
            known.astype(np.float32),
            -1,
            (patch_size, patch_size),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )
        eligible = ~known & (counts > 0.001)
        priority = counts * (1 - counts)
        priority[~eligible] = -1
        if not np.any(eligible):
            raise RuntimeError("No fill front found")
        cy, cx = np.unravel_index(np.argmax(priority), priority.shape)
        y = int(np.clip(cy - half, 0, h - patch_size))
        x = int(np.clip(cx - half, 0, w - patch_size))
        local_known = known[y : y + patch_size, x : x + patch_size]
        template = (
            result[y : y + patch_size, x : x + patch_size].astype(np.float32) / 255
        )
        weights = local_known.astype(np.float32)
        scores = cv2.matchTemplate(src, template, cv2.TM_SQDIFF, mask=weights)
        scores = np.maximum(scores, 0) / max(1, 3 * np.sum(weights))
        scores[~valid] = np.inf
        scores += 0.00035 * uses
        best = float(np.min(scores))
        choices = np.flatnonzero(scores <= best + max(0.0003, best * 0.035))
        index = int(rng.choice(choices))
        dy, dx = np.unravel_index(index, scores.shape)
        donor = source[dy : dy + patch_size, dx : dx + patch_size]
        target = result[y : y + patch_size, x : x + patch_size]
        target[~local_known] = donor[~local_known]
        known[y : y + patch_size, x : x + patch_size] = True
        uses[max(0, dy - half) : dy + half + 1, max(0, dx - half) : dx + half + 1] += 1
        iterations += 1
        if iterations > 2500:
            raise RuntimeError("Fill iteration limit reached")
    assert np.array_equal(result[initial], canvas[initial])
    return result


def three_crops(rgb, mask):
    """Return three grayscale fabric regions, their boxes and foreground coverage."""
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError("Small mask")
    x0, x1, y0, y1 = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
    scale = min(1, 256 / max(h, w))
    sw, sh = (round(w * scale), round(h * scale))
    small = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    integ = cv2.integral(small)
    vertical = y1 - y0 >= x1 - x0
    boxes = []
    tiles = []
    coverage = []
    for fraction in [0.25, 0.5, 0.75]:
        tx = (x0 + x1) / 2 if vertical else x0 + fraction * (x1 - x0)
        ty = y0 + fraction * (y1 - y0) if vertical else (y0 + y1) / 2
        found = None
        maximum = max(8, int(min(x1 - x0, y1 - y0) * 0.7 * scale))
        minimum = max(6, int(min(x1 - x0, y1 - y0) * 0.16 * scale))
        for side in range(maximum, minimum - 1, -2):
            area = (
                integ[side:, side:]
                - integ[:-side, side:]
                - integ[side:, :-side]
                + integ[:-side, :-side]
            )
            yy, xx = np.nonzero(area >= 0.98 * side * side)
            cx = (xx + side / 2) / scale
            cy = (yy + side / 2) / scale
            loc = cy if vertical else cx
            lo = y0 if vertical else x0
            span = y1 - y0 if vertical else x1 - x0
            valid = np.abs((loc - lo) / span - fraction) <= 0.14
            for bx, by, bs in boxes:
                valid &= np.hypot(cx - bx - bs / 2, cy - by - bs / 2) >= 0.28 * min(
                    bs, side / scale
                )
            ids = np.flatnonzero(valid)
            if not len(ids):
                continue
            idx = ids[
                np.argmin(
                    ((cx[ids] - tx) / (x1 - x0)) ** 2
                    + ((cy[ids] - ty) / (y1 - y0)) ** 2
                )
            ]
            size = min(round(side / scale), h, w)
            left = min(max(0, round(xx[idx] / scale)), w - size)
            top = min(max(0, round(yy[idx] / scale)), h - size)
            cov = float(mask[top : top + size, left : left + size].mean())
            if cov < 0.975:
                continue
            found = (left, top, size, cov)
            break
        if found is None:
            raise ValueError("Cannot find three distinct fabric regions")
        x, y, s, cov = found
        boxes.append([int(x), int(y), int(s)])
        coverage.append(cov)
        tiles.append(
            cv2.resize(
                cv2.cvtColor(rgb[y : y + s, x : x + s], cv2.COLOR_RGB2GRAY),
                (128, 128),
                interpolation=cv2.INTER_AREA if s >= 128 else cv2.INTER_LINEAR,
            )
        )
    return (np.stack(tiles), boxes, coverage)
