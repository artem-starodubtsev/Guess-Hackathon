"""One public function for automatic RGB palette extraction; no file IO."""

import cv2
import numpy as np

__all__ = ["extract_palette"]


def extract_palette(
    image: np.ndarray,
    error_threshold: float = 12.0,
    max_colors: int = 4,
    *,
    max_samples: int = 65536,
) -> dict:
    """Choose the smallest K whose palette meets a Lab reconstruction threshold."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy array in RGB format")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have nonempty shape (height, width, 3)")
    for name, value in [("max_colors", max_colors), ("max_samples", max_samples)]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(error_threshold, (bool, np.bool_))
        or not isinstance(error_threshold, (int, float, np.integer, np.floating))
        or (not np.isfinite(error_threshold))
        or (error_threshold < 0)
    ):
        raise ValueError("error_threshold must be a finite nonnegative number")
    pixels = np.ascontiguousarray(image).reshape(-1, 3)
    count = len(pixels)
    sample_count = min(count, int(max_samples))
    if sample_count < count:
        indices = np.random.default_rng(27).choice(count, sample_count, replace=False)
        sample_rgb = pixels[indices]
    else:
        sample_rgb = pixels
    sample_lab = cv2.cvtColor(
        sample_rgb.astype(np.float32).reshape(1, -1, 3) / 255, cv2.COLOR_RGB2LAB
    ).reshape(-1, 3)
    unique_count = len(np.unique(sample_rgb, axis=0))
    candidates = []
    best = None
    for k in range(1, min(int(max_colors), unique_count) + 1):
        if k == 1:
            centers = sample_lab.mean(axis=0, dtype=np.float64).astype(np.float32)[
                None, :
            ]
        else:
            cv2.setRNGSeed(27)
            _, _, centers = cv2.kmeans(
                sample_lab.reshape(-1, 1, 3),
                k,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.03),
                3,
                cv2.KMEANS_PP_CENTERS,
            )
        counts = np.zeros(k, np.int64)
        squared_error = 0.0
        for start in range(0, count, 16384):
            end = min(start + 16384, count)
            if sample_count == count:
                chunk = sample_lab[start:end]
            else:
                chunk = cv2.cvtColor(
                    pixels[start:end].astype(np.float32).reshape(1, -1, 3) / 255,
                    cv2.COLOR_RGB2LAB,
                ).reshape(-1, 3)
            distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            counts += np.bincount(labels, minlength=k)
            squared_error += float(
                np.sum(distances[np.arange(len(chunk)), labels], dtype=np.float64)
            )
        error = float(np.sqrt(squared_error / count))
        candidates.append({"k": k, "rmse_lab": error})
        candidate = {"centers": centers, "counts": counts, "error": error}
        if best is None or error < best["error"]:
            best = candidate
        if error <= error_threshold:
            best = candidate
            break
    active = np.flatnonzero(best["counts"] > 0)
    order = active[np.argsort(-best["counts"][active], kind="stable")]
    centers = best["centers"][order]
    weights = (best["counts"][order] / count).astype(np.float32)
    rgb = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    colors_rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    k = len(weights)
    vector = np.zeros((int(max_colors), 4), np.float32)
    vector[:k, :3] = colors_rgb.astype(np.float32) / 255
    vector[:k, 3] = weights
    return {
        "k": k,
        "colors_rgb": colors_rgb,
        "colors_lab": centers,
        "weights": weights,
        "vector": vector.ravel(),
        "error_lab": best["error"],
        "threshold_met": best["error"] <= error_threshold,
        "error_threshold": float(error_threshold),
        "errors_by_k": candidates,
        "sampled_pixels": sample_count,
        "total_pixels": count,
    }
