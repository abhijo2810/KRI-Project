# ml/utils/patch_infer.py
"""
Sliding-window inference for full-resolution images.

Predictions from overlapping patches are averaged in probability space
before thresholding, which reduces tile-boundary artefacts.
"""

import cv2
import numpy as np
import torch

from ml.datasets.patch_dataset import _patch_starts

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict_sliding_window(
    img_bgr: np.ndarray,
    model,
    device: str,
    patch_size: int = 512,
    overlap: float = 0.25,
    thr: float = 0.5,
    imagenet_norm: bool = False,
) -> np.ndarray:
    """
    Run sliding-window inference over a full-resolution image.

    Returns a binary mask (uint8 0/255) at the input image's resolution.
    """
    h, w   = img_bgr.shape[:2]
    step   = int(patch_size * (1.0 - overlap))
    prob_acc = np.zeros((h, w), dtype=np.float32)
    weight   = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y0 in _patch_starts(h, patch_size, step):
            for x0 in _patch_starts(w, patch_size, step):
                patch = img_bgr[y0:y0 + patch_size, x0:x0 + patch_size]
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if imagenet_norm:
                    patch = (patch - _IMAGENET_MEAN) / _IMAGENET_STD
                x = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                prob_acc[y0:y0 + patch_size, x0:x0 + patch_size] += prob
                weight[y0:y0 + patch_size, x0:x0 + patch_size]   += 1.0

    weight = np.maximum(weight, 1.0)
    return ((prob_acc / weight) > thr).astype(np.uint8) * 255


def dice_iou_full_image(
    img_bgr: np.ndarray,
    gt_mask: np.ndarray,
    model,
    device: str,
    patch_size: int,
    overlap: float,
    thr: float,
    imagenet_norm: bool,
) -> tuple[float, float]:
    """Return (dice, iou) for one full-resolution image."""
    pred = predict_sliding_window(
        img_bgr, model, device, patch_size, overlap, thr, imagenet_norm
    )
    pred_bin = (pred > 127).astype(np.float32)
    gt_bin   = (gt_mask > 0).astype(np.float32)
    eps   = 1e-6
    inter = (pred_bin * gt_bin).sum()
    dice  = (2.0 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps)
    union = (pred_bin + gt_bin - pred_bin * gt_bin).sum()
    iou   = (inter + eps) / (union + eps)
    return float(dice), float(iou)
