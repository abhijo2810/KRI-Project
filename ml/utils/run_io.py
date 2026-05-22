# ml/utils/run_io.py
"""Save a training run to a versioned directory under runs/ and auto-generate summary.md."""

import json
import shutil
from datetime import datetime
from pathlib import Path


def save_run(
    cv_results_path: Path,
    runs_root: Path,
    config: dict,
    extra_plots: list[Path] | None = None,
) -> Path:
    """
    Create a timestamped run directory and populate it.

    Directory name: runs/{YYYY-MM-DD_HHMM}_{model}_{loss}[_patch]

    Contents:
        config.json   — full hyperparameter dict
        metrics.json  — CV summary (subset of cv_results)
        summary.md    — auto-generated human-readable report
        *.png         — copied from outputs/ml/ (threshold sweeps, cover plots)

    Returns the path of the created run directory.
    """
    ts   = datetime.now().strftime("%Y-%m-%d_%H%M")
    name = "_".join(filter(None, [
        ts,
        config.get("model_type", ""),
        config.get("loss_type", ""),
        "patch" if config.get("patch_mode") else "",
    ]))
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── config.json ───────────────────────────────────────────────────────────
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── metrics.json ──────────────────────────────────────────────────────────
    cv = json.loads(cv_results_path.read_text())
    metrics = {
        "mean_val_dice":           cv.get("mean_val_dice"),
        "std_val_dice":            cv.get("std_val_dice"),
        "mean_val_iou":            cv.get("mean_val_iou"),
        "std_val_iou":             cv.get("std_val_iou"),
        "mean_optimal_threshold":  cv.get("mean_optimal_threshold"),
        "cover_pearson_r":         cv.get("cover_correlation", {}).get("pearson_r"),
        "cover_spearman_rho":      cv.get("cover_correlation", {}).get("spearman_rho"),
        "bland_altman_bias":       cv.get("cover_correlation", {}).get("bland_altman", {}).get("bias"),
        "per_fold": [
            {"fold": f["fold"], "dice": f["best_val_dice"], "iou": f["best_val_iou"],
             "opt_thr": f.get("optimal_threshold")}
            for f in cv.get("per_fold", [])
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Copy plots ─────────────────────────────────────────────────────────────
    plots_dir = cv_results_path.parent
    for png in sorted(plots_dir.glob("*.png")):
        shutil.copy2(png, run_dir / png.name)
    if extra_plots:
        for png in extra_plots:
            if png.exists():
                shutil.copy2(png, run_dir / png.name)

    # ── summary.md ────────────────────────────────────────────────────────────
    (run_dir / "summary.md").write_text(_make_summary(name, config, metrics, cv))

    print(f"Run saved to: {run_dir}")
    return run_dir


def _make_summary(name: str, config: dict, metrics: dict, cv: dict) -> str:
    folds = metrics.get("per_fold", [])
    fold_rows = "\n".join(
        f"| {f['fold']+1} | {f['dice']:.4f} | {f['iou']:.4f} | {f['opt_thr']:.2f} |"
        for f in folds
    )
    ba = cv.get("cover_correlation", {}).get("bland_altman", {})

    patch_section = ""
    if config.get("patch_mode"):
        patch_section = f"""
## Patch Config
- Patch size: {config.get('patch_size', '—')} px
- Overlap: {config.get('overlap', '—')}
- Epochs: {config.get('patch_epochs', '—')}
- Min blade px: {config.get('min_blade_px', '—')}
"""

    return f"""# Run: {name}

## Model Config
- Model:      {config.get('model_type', '—')}
- Loss:       {config.get('loss_type', '—')} (α={config.get('tversky_alpha','—')}, β={config.get('tversky_beta','—')}, γ={config.get('tversky_gamma','—')})
- Epochs:     {config.get('epochs', config.get('patch_epochs', '—'))}
- LR:         {config.get('lr', '—')}
- Batch size: {config.get('batch_size', '—')}
- ImageNet norm: {config.get('imagenet_norm', '—')}
- CLAHE:      {config.get('use_clahe', '—')}
{patch_section}
## CV Results ({config.get('n_folds', 5)}-fold stratified)

| Fold | Dice | IoU | Opt thr |
|------|------|-----|---------|
{fold_rows}

**Mean Dice = {metrics['mean_val_dice']:.4f} ± {metrics['std_val_dice']:.4f}**
**Mean IoU  = {metrics['mean_val_iou']:.4f} ± {metrics['std_val_iou']:.4f}**
Mean optimal threshold = {metrics['mean_optimal_threshold']:.3f}

## Cover Correlation (n={cv.get('cover_correlation',{}).get('n','—')})
- Pearson r   = {metrics.get('cover_pearson_r', '—')}
- Spearman ρ  = {metrics.get('cover_spearman_rho', '—')}
- Bland-Altman: bias = {ba.get('bias','—')}%  LoA [{ba.get('lo_loa','—')}, {ba.get('hi_loa','—')}]
"""
