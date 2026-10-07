"""Shared white-background segmentation for pattern and shape extraction."""

import cv2
import numpy as np


def _garment_mask(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    near_bg = (np.linalg.norm(lab - bg, axis=2) < 24).astype(np.uint8)
    count, labels = cv2.connectedComponents(near_bg, connectivity=8)
    edge_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    edge_labels = edge_labels[edge_labels != 0]
    foreground = (~np.isin(labels, edge_labels)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No garment found")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    if np.mean(mask > 0) > 0.94 or cv2.contourArea(contour) < 100:
        raise ValueError("No reliable separation from white background")
    trimap = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    trimap[mask > 0] = cv2.GC_PR_FGD
    trimap[(mask > 0) & (near_bg > 0)] = cv2.GC_PR_BGD
    interior = cv2.erode(mask, np.ones((7, 7), np.uint8))
    trimap[(interior > 0) & (near_bg == 0)] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(
        image,
        trimap,
        None,
        np.zeros((1, 65), np.float64),
        np.zeros((1, 65), np.float64),
        3,
        cv2.GC_INIT_WITH_MASK,
    )
    mask = np.where((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    return mask
