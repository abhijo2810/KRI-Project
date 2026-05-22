# ml/utils/threshold_sweep.py
"""Threshold sweep over cached softmax probabilities."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_threshold_sweep(
    all_probs: list,
    all_targets: list,
    out_dir: Path,
    fold_k: int,
) -> dict:
    """
    Sweep thresholds 0.10–0.90 in 0.05 steps over pre-computed probabilities.

    Args:
        all_probs:   List of (H, W) float32 arrays in [0, 1].
        all_targets: List of (H, W) float32 arrays in {0, 1}.
        out_dir:     Directory to write CSV and PNG.
        fold_k:      Fold index (used in filenames and plot titles).

    Returns:
        dict with optimal_threshold, best_dice, best_iou, and per-threshold records.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(0.10, 0.91, 0.05)
    eps = 1e-6

    records = []
    for thr in thresholds:
        dice_scores, iou_scores = [], []
        for prob, target in zip(all_probs, all_targets):
            pred  = (prob > thr).astype(np.float32)
            inter = (pred * target).sum()
            d     = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
            union = pred.sum() + target.sum() - inter
            i     = (inter + eps) / (union + eps)
            dice_scores.append(float(d))
            iou_scores.append(float(i))
        records.append({
            "threshold": round(float(thr), 3),
            "dice":      round(float(np.mean(dice_scores)), 6),
            "iou":       round(float(np.mean(iou_scores)),  6),
        })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / f"threshold_sweep_fold_{fold_k}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "dice", "iou"])
        writer.writeheader()
        writer.writerows(records)

    # ── Plot ──────────────────────────────────────────────────────────────────
    thrs   = [r["threshold"] for r in records]
    dices  = [r["dice"] for r in records]
    ious   = [r["iou"]  for r in records]
    best_i = int(np.argmax(dices))
    best_t = thrs[best_i]
    best_d = dices[best_i]
    best_u = ious[best_i]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, dices, "b-o", markersize=5, label="Dice")
    ax.plot(thrs, ious,  "r-s", markersize=5, label="IoU")
    ax.axvline(best_t, color="steelblue", linestyle="--", alpha=0.7,
               label=f"Best thr = {best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Threshold sweep — Fold {fold_k + 1}  "
        f"(best Dice = {best_d:.4f} @ {best_t:.2f})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"threshold_sweep_fold_{fold_k}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {
        "optimal_threshold": best_t,
        "best_dice": best_d,
        "best_iou":  best_u,
        "csv_path":  str(csv_path),
        "png_path":  str(png_path),
        "records":   records,
    }
