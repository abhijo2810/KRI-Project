# ml/train_unet.py
"""
Training entry point for bryozoan segmentation.

Single split:  python -m ml.train_unet
5-fold CV:     python -m ml.train_unet --cv

──────────────────────────────────────────────────
CONFIG — edit the block below to change behaviour.
──────────────────────────────────────────────────
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from ml.models.model_factory import build_model, uses_imagenet_norm, select_device
from ml.datasets.bryozoan_dataset import BryozoanDataset
from ml.utils.losses import dice_bce_loss, tversky_loss, focal_tversky_loss
from ml.utils.metrics import batch_dice, batch_iou
from ml.utils.threshold_sweep import run_threshold_sweep
from ml.utils.cover_metrics import save_cover_analysis, _cover_pct

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
MODEL_TYPE    = "smp_resnet18"  # "custom_unet" | "smp_resnet18" | "smp_effnet_b0"
BATCH_SIZE    = 2
EPOCHS        = 60
LR            = 1e-3
VAL_FRAC      = 0.2             # used only in single-split mode
INPUT_SIZE    = 512
BATCH_NAME    = "bryozoan_batch_01"
N_FOLDS       = 5
USE_CLAHE     = False           # add CLAHE to augmentation pipeline
LOSS_TYPE     = "dice_bce"      # "dice_bce" | "tversky" | "focal_tversky"
TVERSKY_ALPHA = 0.3             # FP weight
TVERSKY_BETA  = 0.7             # FN weight
TVERSKY_GAMMA = 0.75            # focal exponent (focal_tversky only)
# ════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_label(gt_path: Path, blade_path: Path | None) -> str:
    """
    Derive a coverage category from a GT mask.

    Thresholds chosen so this batch yields 5 clean / 6 light / 7 medium / 6 heavy.
    """
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return "unknown"
    bry_px = (gt > 0).sum()
    if bry_px == 0:
        return "clean"
    # Use blade mask as denominator when available
    if blade_path is not None and blade_path.exists():
        blade = cv2.imread(str(blade_path), cv2.IMREAD_GRAYSCALE)
        denom = int((blade > 0).sum()) if blade is not None else gt.size
    else:
        denom = gt.size
    cov = 100.0 * bry_px / max(denom, 1)
    if cov > 65.0:
        return "heavy"
    if cov > 18.0:
        return "medium"
    return "light"


def _get_labels(images: list, masks_dir: Path, blade_dir: Path) -> list:
    labels = []
    for p in images:
        gt_path    = masks_dir / f"{p.stem}_bry_gt.png"
        blade_path = blade_dir / f"{p.stem}_blade_mask.png"
        labels.append(_coverage_label(gt_path, blade_path))
    return labels


def _make_datasets(images_dir, masks_dir, train_idx, val_idx, imagenet_norm):
    kwargs = dict(images_dir=images_dir, masks_dir=masks_dir,
                  size=INPUT_SIZE, imagenet_norm=imagenet_norm, use_clahe=USE_CLAHE)
    train_ds = BryozoanDataset(**kwargs, augment=True,  indices=train_idx, rng_seed=0)
    val_ds   = BryozoanDataset(**kwargs, augment=False, indices=val_idx)
    return train_ds, val_ds


def _evaluate_checkpoint(
    ckpt_path: Path,
    images_dir: Path,
    masks_dir: Path,
    blade_dir: Path,
    val_idx: list,
    imagenet_norm: bool,
    optimal_thr: float = 0.5,
) -> dict:
    """
    Load best checkpoint, run inference on val set, collect probs, targets,
    and percent-cover pairs. Returns dict ready for threshold sweep + cover analysis.
    """
    cfg_path = ckpt_path.with_suffix(".json")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {
        "model_type": MODEL_TYPE, "input_size": INPUT_SIZE, "imagenet_norm": imagenet_norm
    }

    model, device = select_device(build_model(cfg["model_type"]))
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    val_ds = BryozoanDataset(
        images_dir, masks_dir=masks_dir, size=INPUT_SIZE,
        augment=False, indices=val_idx, imagenet_norm=imagenet_norm,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    all_probs, all_targets, all_stems = [], [], []
    pred_covers, gt_covers = [], []

    with torch.no_grad():
        for b in val_dl:
            if "mask" not in b:
                continue
            x   = b["image"].to(device)
            y   = b["mask"]          # (1, 1, H, W) stays CPU
            stem = b["stem"][0]

            logits = model(x)
            prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()   # (H, W)
            target = y[0, 0].numpy()                             # (H, W) {0,1}

            all_probs.append(prob)
            all_targets.append(target)
            all_stems.append(stem)

            # Percent cover — intersect with blade mask if available
            blade_path = blade_dir / f"{stem}_blade_mask.png"
            if blade_path.exists():
                blade = cv2.imread(str(blade_path), cv2.IMREAD_GRAYSCALE)
                blade_bin = cv2.resize(
                    blade, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST
                ) if blade is not None else None
                blade_bin = (blade_bin > 0) if blade_bin is not None else None
            else:
                blade_bin = None

            pred_bin = (prob > optimal_thr).astype(np.float32)
            pred_covers.append(_cover_pct(pred_bin, blade_bin))
            gt_covers.append(_cover_pct(target,   blade_bin))

    return {
        "probs":       all_probs,
        "targets":     all_targets,
        "stems":       all_stems,
        "pred_covers": pred_covers,
        "gt_covers":   gt_covers,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if LOSS_TYPE == "dice_bce":
        return dice_bce_loss(logits, targets, bce_weight=0.5)
    if LOSS_TYPE == "tversky":
        return tversky_loss(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    if LOSS_TYPE == "focal_tversky":
        return focal_tversky_loss(logits, targets,
                                  alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
                                  gamma=TVERSKY_GAMMA)
    raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE!r}")


def train_one_run(
    images_dir: Path,
    masks_dir: Path,
    train_idx: list,
    val_idx: list,
    ckpt_path: Path,
    imagenet_norm: bool,
    label: str = "",
) -> dict:
    """Train for EPOCHS epochs; save best checkpoint by val Dice."""
    train_ds, val_ds = _make_datasets(images_dir, masks_dir, train_idx, val_idx, imagenet_norm)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, device = select_device(build_model(MODEL_TYPE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, min_lr=1e-6
    )

    best_dice, best_iou = -1.0, 0.0
    tag = f"[{label}] " if label else ""

    print(f"\n{tag}Model: {MODEL_TYPE}  Device: {device}  "
          f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValDice':>8}  {'ValIoU':>7}")
    print("─" * 52)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for b in train_dl:
            if "mask" not in b:
                continue
            x = b["image"].to(device)
            y = b["mask"].to(device)
            logits = model(x)
            loss   = _loss(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item(); t_n += 1
        avg_train = t_loss / max(t_n, 1)

        model.eval()
        v_loss = v_dice = v_iou = 0.0; v_n = 0
        with torch.no_grad():
            for b in val_dl:
                if "mask" not in b:
                    continue
                x = b["image"].to(device)
                y = b["mask"].to(device)
                logits = model(x)
                v_loss += _loss(logits, y).item()
                v_dice += batch_dice(logits, y)
                v_iou  += batch_iou(logits, y)
                v_n    += 1

        avg_val_loss = v_loss / max(v_n, 1)
        avg_val_dice = v_dice / max(v_n, 1)
        avg_val_iou  = v_iou  / max(v_n, 1)

        print(f"{epoch:5d}  {avg_train:10.4f}  {avg_val_loss:9.4f}  "
              f"{avg_val_dice:8.4f}  {avg_val_iou:7.4f}")

        scheduler.step(avg_val_dice)

        if avg_val_dice > best_dice and v_n > 0:
            best_dice = avg_val_dice
            best_iou  = avg_val_iou
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            cfg = {"model_type": MODEL_TYPE, "input_size": INPUT_SIZE,
                   "imagenet_norm": imagenet_norm}
            ckpt_path.with_suffix(".json").write_text(json.dumps(cfg, indent=2))
            print(f"  -> {tag}Best Dice={best_dice:.4f} — saved {ckpt_path.name}")

    return {"best_val_dice": best_dice, "best_val_iou": best_iou}


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(images_dir: Path, masks_dir: Path, out_name: str | None = None) -> None:
    blade_dir = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"
    out_dir   = ROOT / "outputs" / "ml"
    ckpt_dir  = ROOT / "checkpoints" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build full dataset to get the canonical image list
    full_ds = BryozoanDataset(images_dir, masks_dir=masks_dir, size=INPUT_SIZE)
    n = len(full_ds)
    if n == 0:
        print("[INFO] No usable GT mask pairs found."); return
    print(f"[INFO] Total labeled pairs: {n}  ({N_FOLDS}-fold CV)\n")

    images = full_ds.images
    labels = _get_labels(images, masks_dir, blade_dir)
    label_counts = {l: labels.count(l) for l in sorted(set(labels))}
    print(f"[INFO] Coverage distribution: {label_counts}")

    imagenet_norm = uses_imagenet_norm(MODEL_TYPE)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    # Aggregate across all folds for Task 4
    all_pred_covers, all_gt_covers, all_stems = [], [], []

    for fold_k, (train_np, val_np) in enumerate(skf.split(images, labels)):
        train_idx = train_np.tolist()
        val_idx   = val_np.tolist()

        print(f"\n{'═'*60}")
        print(f"  FOLD {fold_k + 1} / {N_FOLDS}   "
              f"train={len(train_idx)}  val={len(val_idx)}")
        val_labels = [labels[i] for i in val_idx]
        print(f"  Val categories: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
        print(f"{'═'*60}")

        ckpt_path = ckpt_dir / f"fold_{fold_k}.pt"

        # ── Train ─────────────────────────────────────────────────────────────
        train_result = train_one_run(
            images_dir, masks_dir, train_idx, val_idx,
            ckpt_path, imagenet_norm, label=f"fold {fold_k+1}/{N_FOLDS}",
        )

        # ── Evaluate best checkpoint at thr=0.5 to collect probs + covers ────
        eval_data = _evaluate_checkpoint(
            ckpt_path, images_dir, masks_dir, blade_dir,
            val_idx, imagenet_norm, optimal_thr=0.5,
        )

        # ── Threshold sweep (Task 3) ──────────────────────────────────────────
        sweep = run_threshold_sweep(
            eval_data["probs"], eval_data["targets"], out_dir, fold_k
        )
        opt_thr = sweep["optimal_threshold"]
        print(f"\n  [Fold {fold_k+1}] Optimal threshold = {opt_thr:.2f}  "
              f"(Dice={sweep['best_dice']:.4f}  IoU={sweep['best_iou']:.4f})")

        # Re-collect covers with the optimal threshold for Task 4
        eval_data_opt = _evaluate_checkpoint(
            ckpt_path, images_dir, masks_dir, blade_dir,
            val_idx, imagenet_norm, optimal_thr=opt_thr,
        )
        all_pred_covers.extend(eval_data_opt["pred_covers"])
        all_gt_covers.extend(eval_data_opt["gt_covers"])
        all_stems.extend(eval_data_opt["stems"])

        fold_results.append({
            "fold": fold_k,
            "val_stems": [images[i].stem for i in val_idx],
            "val_categories": {str(k): int(v) for k, v in
                               zip(*np.unique(val_labels, return_counts=True))},
            "best_val_dice":      round(train_result["best_val_dice"], 4),
            "best_val_iou":       round(train_result["best_val_iou"],  4),
            "optimal_threshold":  opt_thr,
            "sweep_dice_at_opt":  round(sweep["best_dice"], 4),
            "sweep_iou_at_opt":   round(sweep["best_iou"],  4),
        })

    # ── CV summary ────────────────────────────────────────────────────────────
    dice_vals = [r["best_val_dice"]  for r in fold_results]
    iou_vals  = [r["best_val_iou"]   for r in fold_results]
    thr_vals  = [r["optimal_threshold"] for r in fold_results]
    mean_thr  = float(np.mean(thr_vals))

    print(f"\n{'═'*60}")
    print(f"  CROSS-VALIDATION SUMMARY ({MODEL_TYPE}  clahe={USE_CLAHE})")
    print(f"{'─'*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}:  Dice={r['best_val_dice']:.4f}  "
              f"IoU={r['best_val_iou']:.4f}  opt_thr={r['optimal_threshold']:.2f}")
    print(f"{'─'*60}")
    print(f"  Mean Dice = {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    print(f"  Mean IoU  = {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
    print(f"  Mean opt threshold = {mean_thr:.3f}")
    print(f"{'═'*60}\n")

    # ── Cover analysis (Task 4) ────────────────────────────────────────────────
    cover_result = save_cover_analysis(all_pred_covers, all_gt_covers, all_stems, out_dir)

    # ── Save cv_results.json ──────────────────────────────────────────────────
    cv_summary = {
        "model_type":            MODEL_TYPE,
        "loss_type":             LOSS_TYPE,
        "tversky_alpha":         TVERSKY_ALPHA,
        "tversky_beta":          TVERSKY_BETA,
        "tversky_gamma":         TVERSKY_GAMMA,
        "epochs":                EPOCHS,
        "n_folds":               N_FOLDS,
        "use_clahe":             USE_CLAHE,
        "mean_val_dice":         round(float(np.mean(dice_vals)), 4),
        "std_val_dice":          round(float(np.std(dice_vals)),  4),
        "mean_val_iou":          round(float(np.mean(iou_vals)),  4),
        "std_val_iou":           round(float(np.std(iou_vals)),   4),
        "mean_optimal_threshold": round(mean_thr, 3),
        "per_fold":              fold_results,
        "cover_correlation":     cover_result,
    }
    suffix    = out_name or LOSS_TYPE
    json_path = out_dir / f"cv_results_{suffix}.json"
    json_path.write_text(json.dumps(cv_summary, indent=2))
    print(f"Results saved to: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train bryozoan segmentation model")
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold stratified cross-validation")
    parser.add_argument("--loss", default=None,
                        choices=["dice_bce", "tversky", "focal_tversky"],
                        help="Override LOSS_TYPE config for this run")
    parser.add_argument("--out-name", default=None,
                        help="Suffix for cv_results_<name>.json (default: LOSS_TYPE)")
    parser.add_argument("--save-run", action="store_true",
                        help="Save results to a timestamped runs/ directory (Task 8)")
    args = parser.parse_args()

    # CLI overrides config-block values
    global LOSS_TYPE
    if args.loss:
        LOSS_TYPE = args.loss

    images_dir = ROOT / "data" / "raw_for_annot"  / BATCH_NAME
    masks_dir  = ROOT / "data" / "processed_annot" / BATCH_NAME / "bryozoan_gt_masks"

    if not masks_dir.exists():
        print(f"[INFO] GT masks folder missing: {masks_dir}")
        print("Run scripts/convert_goodnotes_annotations.py first.")
        return

    if args.cv:
        run_cv(images_dir, masks_dir, out_name=args.out_name)
        if args.save_run:
            from ml.utils.run_io import save_run
            suffix    = args.out_name or LOSS_TYPE
            cv_path   = ROOT / "outputs" / "ml" / f"cv_results_{suffix}.json"
            run_cfg   = {
                "model_type": MODEL_TYPE, "loss_type": LOSS_TYPE,
                "tversky_alpha": TVERSKY_ALPHA, "tversky_beta": TVERSKY_BETA,
                "tversky_gamma": TVERSKY_GAMMA, "epochs": EPOCHS, "lr": LR,
                "batch_size": BATCH_SIZE, "input_size": INPUT_SIZE,
                "imagenet_norm": uses_imagenet_norm(MODEL_TYPE),
                "use_clahe": USE_CLAHE, "n_folds": N_FOLDS,
            }
            save_run(cv_path, ROOT / "runs", run_cfg)
        return

    # ── Single split (original behaviour) ────────────────────────────────────
    full_ds = BryozoanDataset(images_dir, masks_dir=masks_dir, size=INPUT_SIZE)
    n = len(full_ds)
    if n == 0:
        print("[INFO] No usable GT mask pairs found."); return
    print(f"[INFO] Total labeled pairs: {n}")

    rng       = np.random.default_rng(seed=42)
    idx       = rng.permutation(n).tolist()
    n_val     = max(1, int(n * VAL_FRAC))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    imagenet_norm = uses_imagenet_norm(MODEL_TYPE)
    ckpt_path     = ROOT / "models" / "bry_unet.pt"

    result = train_one_run(
        images_dir, masks_dir, train_idx, val_idx,
        ckpt_path, imagenet_norm,
    )
    print(f"\n[DONE] Best val Dice={result['best_val_dice']:.4f}  "
          f"IoU={result['best_val_iou']:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
