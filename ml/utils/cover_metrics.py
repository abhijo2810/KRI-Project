# ml/utils/cover_metrics.py
"""
Percent-cover correlation analysis between predicted and GT bryozoan masks.

Outputs:
  cover_correlation.png  — scatter plot of pred vs GT cover with y=x line
  bland_altman_cover.png — Bland-Altman agreement plot
  cover_validation.json  — Pearson r, Spearman ρ, and per-image data
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _cover_pct(binary_mask: np.ndarray, blade_mask: np.ndarray | None) -> float:
    """
    Bryozoan coverage as % of blade area.
    Falls back to total image pixels when blade mask is unavailable.
    """
    bry_px = binary_mask.sum()
    denom  = blade_mask.sum() if blade_mask is not None and blade_mask.sum() > 0 else binary_mask.size
    return 100.0 * bry_px / max(denom, 1)


def save_cover_analysis(
    pred_covers: list,
    gt_covers: list,
    stems: list,
    out_dir: Path,
) -> dict:
    """
    Run Pearson/Spearman correlation, Bland-Altman, and scatter plots.

    Args:
        pred_covers: Predicted % cover per image.
        gt_covers:   GT % cover per image.
        stems:       Corresponding image stems (for the JSON output).
        out_dir:     Directory to write plots and JSON.

    Returns:
        dict with all metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.array(pred_covers, dtype=float)
    gt   = np.array(gt_covers,   dtype=float)

    pearson_r, pearson_p   = stats.pearsonr(pred, gt)
    spearman_r, spearman_p = stats.spearmanr(pred, gt)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, color="steelblue", alpha=0.8, edgecolors="white", s=60)
    lim = max(gt.max(), pred.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("GT bryozoan cover (%)")
    ax.set_ylabel("Predicted bryozoan cover (%)")
    ax.set_title(
        f"Percent cover correlation\n"
        f"Pearson r = {pearson_r:.3f}  Spearman ρ = {spearman_r:.3f}"
    )
    ax.legend()
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    scatter_path = out_dir / "cover_correlation.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # ── Bland-Altman ──────────────────────────────────────────────────────────
    means  = (pred + gt) / 2.0
    diffs  = pred - gt
    bias   = diffs.mean()
    sd     = diffs.std(ddof=1)
    lo_loa = bias - 1.96 * sd
    hi_loa = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, color="steelblue", alpha=0.8, edgecolors="white", s=60)
    ax.axhline(bias,   color="black",  linewidth=1.5, label=f"Bias = {bias:+.2f}%")
    ax.axhline(hi_loa, color="red",    linewidth=1,   linestyle="--",
               label=f"+1.96 SD = {hi_loa:+.2f}%")
    ax.axhline(lo_loa, color="red",    linewidth=1,   linestyle="--",
               label=f"−1.96 SD = {lo_loa:+.2f}%")
    ax.set_xlabel("Mean of predicted and GT cover (%)")
    ax.set_ylabel("Predicted − GT cover (%)")
    ax.set_title("Bland-Altman: predicted vs GT bryozoan cover")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ba_path = out_dir / "bland_altman_cover.png"
    fig.savefig(ba_path, dpi=150)
    plt.close(fig)

    # ── JSON ─────────────────────────────────────────────────────────────────
    result = {
        "n":             len(pred),
        "pearson_r":     round(float(pearson_r),  4),
        "pearson_p":     round(float(pearson_p),  6),
        "spearman_rho":  round(float(spearman_r), 4),
        "spearman_p":    round(float(spearman_p), 6),
        "bland_altman":  {
            "bias":   round(float(bias),   3),
            "sd":     round(float(sd),     3),
            "lo_loa": round(float(lo_loa), 3),
            "hi_loa": round(float(hi_loa), 3),
        },
        "per_image": [
            {"stem": s, "gt_cover": round(float(g), 3), "pred_cover": round(float(p), 3)}
            for s, g, p in zip(stems, gt_covers, pred_covers)
        ],
    }
    json_path = out_dir / "cover_validation.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"\nCover correlation  n={len(pred)}")
    print(f"  Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:.4f}  (p={spearman_p:.4f})")
    print(f"  Bland-Altman bias = {bias:+.2f}%  LoA [{lo_loa:.2f}, {hi_loa:.2f}]")
    print(f"  Saved: {scatter_path.name}  {ba_path.name}  {json_path.name}")

    return result
