#!/usr/bin/env python3
# compare_runs.py
"""
Scan runs/ and print a ranked comparison table of all training runs.

Usage:
    python compare_runs.py
    python compare_runs.py --runs-dir /path/to/runs
    python compare_runs.py --sort iou
"""
import argparse
import json
from pathlib import Path


def _read_run(run_dir: Path) -> dict | None:
    cfg_p  = run_dir / "config.json"
    met_p  = run_dir / "metrics.json"
    if not cfg_p.exists() or not met_p.exists():
        return None
    cfg = json.loads(cfg_p.read_text())
    met = json.loads(met_p.read_text())
    return {
        "name":       run_dir.name,
        "model":      cfg.get("model_type", "—"),
        "loss":       cfg.get("loss_type",  "—"),
        "patch":      "yes" if cfg.get("patch_mode") else "no",
        "clahe":      "yes" if cfg.get("use_clahe")  else "no",
        "epochs":     cfg.get("epochs", cfg.get("patch_epochs", "—")),
        "dice":       met.get("mean_val_dice",  0.0),
        "dice_std":   met.get("std_val_dice",   0.0),
        "iou":        met.get("mean_val_iou",   0.0),
        "opt_thr":    met.get("mean_optimal_threshold", "—"),
        "pearson_r":  met.get("cover_pearson_r",  "—"),
        "ba_bias":    met.get("bland_altman_bias","—"),
    }


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs",
                        help="Path to runs directory (default: ./runs)")
    parser.add_argument("--sort", default="dice",
                        choices=["dice", "iou", "pearson_r", "name"],
                        help="Column to sort by (default: dice)")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        print(f"[INFO] No runs directory found at: {runs_root}")
        print("Run a training experiment with --save-run to create one.")
        return

    rows = []
    for d in sorted(runs_root.iterdir()):
        if d.is_dir():
            r = _read_run(d)
            if r:
                rows.append(r)

    if not rows:
        print(f"[INFO] No valid runs found in: {runs_root}")
        return

    # Sort
    key = args.sort
    rows.sort(key=lambda r: r.get(key, 0.0) if isinstance(r.get(key), float) else 0.0,
              reverse=(key != "name"))

    # Print
    cols = [
        ("Rank", 5), ("Run name", 38), ("Model", 14), ("Loss", 14),
        ("Patch", 6), ("Ep", 4), ("Dice±Std", 13), ("IoU", 7),
        ("OptThr", 7), ("PearsonR", 9), ("BA bias", 8),
    ]
    sep  = "  "
    hdr  = sep.join(f"{c:<{w}}" for c, w in cols)
    line = sep.join("─" * w for _, w in cols)
    print("\n" + hdr)
    print(line)

    for rank, r in enumerate(rows, 1):
        dice_str = f"{_fmt(r['dice'])}±{_fmt(r['dice_std'])}"
        vals = [
            str(rank),
            r["name"][:38],
            r["model"][:14],
            r["loss"][:14],
            r["patch"],
            str(r["epochs"]),
            dice_str,
            _fmt(r["iou"]),
            _fmt(r["opt_thr"]),
            _fmt(r["pearson_r"]) if isinstance(r["pearson_r"], float) else str(r["pearson_r"]),
            _fmt(r["ba_bias"])   if isinstance(r["ba_bias"],   float) else str(r["ba_bias"]),
        ]
        print(sep.join(f"{v:<{w}}" for v, (_, w) in zip(vals, cols)))
    print()


if __name__ == "__main__":
    main()
