# ml/train_patch.py
"""
Patch-based training entry point (Task 7).

Tiles full-resolution images into 512×512 patches (25% overlap),
filters by blade-mask intersection, trains a segmentation model,
and validates on full-resolution images via sliding-window inference.

CV split is stratified by SOURCE IMAGE — patches from the same image
always land on the same side of the fold boundary.

Run from project root:
    python -m ml.train_patch --cv

Condition note (from task spec): implement when Dice < 0.82.
Current best: focal_tversky CV Dice = 0.835 ≥ 0.82.
Patch mode is implemented as specified but may yield marginal gains.

──────────────────────────────────────────────────
CONFIG — edit the block below to change behaviour.
──────────────────────────────────────────────────
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from ml.models.model_factory import build_model, uses_imagenet_norm, select_device
from ml.datasets.patch_dataset import PatchDataset
from ml.utils.losses import dice_bce_loss, tversky_loss, focal_tversky_loss
from ml.utils.patch_infer import dice_iou_full_image
from ml.utils.threshold_sweep import run_threshold_sweep
from ml.utils.cover_metrics import save_cover_analysis, _cover_pct

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
MODEL_TYPE    = "smp_resnet18"
LOSS_TYPE     = "focal_tversky"
TVERSKY_ALPHA = 0.3
TVERSKY_BETA  = 0.7
TVERSKY_GAMMA = 0.75
BATCH_SIZE    = 2
PATCH_EPOCHS  = 20             # fewer than standard; pretrained encoder converges fast
PATCH_LR      = 5e-4           # slightly lower LR for full-res features
PATCH_SIZE    = 512
OVERLAP       = 0.25
MIN_BLADE_PX  = 1000           # min blade pixels to keep a patch
USE_CLAHE     = False
BATCH_NAME    = "bryozoan_batch_01"
N_FOLDS       = 5
# ════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_label(gt_path: Path, blade_path: Path | None) -> str:
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return "unknown"
    bry_px = (gt > 0).sum()
    if bry_px == 0:
        return "clean"
    if blade_path and blade_path.exists():
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


def _loss(logits, targets):
    if LOSS_TYPE == "focal_tversky":
        return focal_tversky_loss(logits, targets,
                                  alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
                                  gamma=TVERSKY_GAMMA)
    if LOSS_TYPE == "tversky":
        return tversky_loss(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA)
    return dice_bce_loss(logits, targets, bce_weight=0.5)


def _validate_full_images(
    model, device, val_paths, masks_dir, imagenet_norm, thr=0.5
) -> tuple[float, float, list, list]:
    """Sliding-window inference on each val image; returns mean (dice, iou) and cover lists."""
    dice_scores, iou_scores = [], []
    pred_covers, gt_covers, stems = [], [], []

    blade_dir = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"

    for p in val_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        gt_path = masks_dir / f"{p.stem}_bry_gt.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue

        d, u = dice_iou_full_image(img_bgr, gt, model, device,
                                    PATCH_SIZE, OVERLAP, thr, imagenet_norm)
        dice_scores.append(d)
        iou_scores.append(u)

        # Cover metrics
        bp = blade_dir / f"{p.stem}_blade_mask.png"
        blade = cv2.imread(str(bp), cv2.IMREAD_GRAYSCALE) if bp.exists() else None
        blade_bin = (blade > 0) if blade is not None else None

        from ml.utils.patch_infer import predict_sliding_window
        pred_mask = predict_sliding_window(img_bgr, model, device,
                                            PATCH_SIZE, OVERLAP, thr, imagenet_norm)
        pred_covers.append(_cover_pct((pred_mask > 127).astype(float), blade_bin))
        gt_covers.append(  _cover_pct((gt > 0).astype(float),          blade_bin))
        stems.append(p.stem)

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou  = float(np.mean(iou_scores))  if iou_scores  else 0.0
    return mean_dice, mean_iou, pred_covers, gt_covers, stems


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_fold(train_paths, val_paths, masks_dir, ckpt_path, imagenet_norm, label=""):
    train_ds = PatchDataset(
        train_paths, masks_dir,
        blade_dir=ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks",
        patch_size=PATCH_SIZE, overlap=OVERLAP, min_blade_px=MIN_BLADE_PX,
        augment=True, imagenet_norm=imagenet_norm, use_clahe=USE_CLAHE, rng_seed=0,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model, device = select_device(build_model(MODEL_TYPE))
    optimizer = torch.optim.Adam(model.parameters(), lr=PATCH_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
    )

    tag       = f"[{label}] " if label else ""
    best_dice = -1.0
    best_iou  = 0.0

    print(f"\n{tag}Model: {MODEL_TYPE}  Loss: {LOSS_TYPE}  Device: {device}")
    print(f"  Train patches: {len(train_ds)}  Val images: {len(val_paths)}")
    print(f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValDice':>8}  {'ValIoU':>7}")
    print("─" * 40)

    for epoch in range(1, PATCH_EPOCHS + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for b in train_dl:
            if "mask" not in b:
                continue
            x = b["image"].to(device)
            y = b["mask"].to(device)
            loss = _loss(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item(); t_n += 1

        avg_train = t_loss / max(t_n, 1)
        val_dice, val_iou, _, _, _ = _validate_full_images(
            model, device, val_paths, masks_dir, imagenet_norm
        )

        print(f"{epoch:5d}  {avg_train:10.4f}  {val_dice:8.4f}  {val_iou:7.4f}")
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou  = val_iou
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            cfg = {"model_type": MODEL_TYPE, "input_size": PATCH_SIZE,
                   "imagenet_norm": imagenet_norm, "patch_mode": True,
                   "patch_size": PATCH_SIZE, "overlap": OVERLAP}
            ckpt_path.with_suffix(".json").write_text(json.dumps(cfg, indent=2))
            print(f"  -> {tag}Best Dice={best_dice:.4f} — saved {ckpt_path.name}")

    return model, device, imagenet_norm, {"best_val_dice": best_dice, "best_val_iou": best_iou}


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv_patch(images_dir: Path, masks_dir: Path, out_name: str | None = None) -> None:
    blade_dir  = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"
    out_dir    = ROOT / "outputs" / "ml"
    ckpt_dir   = ROOT / "checkpoints" / "cv_patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from ml.datasets.bryozoan_dataset import BryozoanDataset
    full_ds = BryozoanDataset(images_dir, masks_dir=masks_dir, size=PATCH_SIZE)
    if len(full_ds) == 0:
        print("[INFO] No GT mask pairs found."); return
    images = full_ds.images

    labels = []
    for p in images:
        labels.append(_coverage_label(
            masks_dir / f"{p.stem}_bry_gt.png",
            blade_dir / f"{p.stem}_blade_mask.png"
        ))

    imagenet_norm = uses_imagenet_norm(MODEL_TYPE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    all_pred_covers, all_gt_covers, all_stems = [], [], []

    print(f"\n[INFO] Patch CV — {len(images)} images  patch_size={PATCH_SIZE}  "
          f"overlap={OVERLAP}  epochs={PATCH_EPOCHS}")

    for fold_k, (train_np, val_np) in enumerate(skf.split(images, labels)):
        train_paths = [images[i] for i in train_np]
        val_paths   = [images[i] for i in val_np]

        print(f"\n{'═'*55}")
        print(f"  FOLD {fold_k+1}/{N_FOLDS}  train={len(train_paths)} images  val={len(val_paths)} images")
        print(f"{'═'*55}")

        ckpt_path = ckpt_dir / f"fold_{fold_k}.pt"
        model, device, _, train_res = _train_fold(
            train_paths, val_paths, masks_dir, ckpt_path, imagenet_norm,
            label=f"fold {fold_k+1}/{N_FOLDS}"
        )

        # Collect probabilities at 0.5 for threshold sweep
        # Re-run val inference at thr=0.5 to get per-image probs for sweep
        # (simplified: collect at image level using sliding window)
        all_probs_fold, all_targets_fold = [], []
        for p in val_paths:
            img_bgr = cv2.imread(str(p))
            gt      = cv2.imread(str(masks_dir / f"{p.stem}_bry_gt.png"),
                                  cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt is None:
                continue
            from ml.utils.patch_infer import predict_sliding_window
            # Get full-res probability map by loading the saved checkpoint
            # and running with thr=0 (collect raw prob)
            # Re-load best checkpoint
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            from ml.utils.patch_infer import _patch_starts
            from ml.datasets.patch_dataset import _patch_starts as ps_fn
            h, w = img_bgr.shape[:2]
            step = int(PATCH_SIZE * (1 - OVERLAP))
            prob_acc = np.zeros((h, w), dtype=np.float32)
            weight   = np.zeros((h, w), dtype=np.float32)
            with torch.no_grad():
                for y0 in ps_fn(h, PATCH_SIZE, step):
                    for x0 in ps_fn(w, PATCH_SIZE, step):
                        patch = img_bgr[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                        if imagenet_norm:
                            from ml.utils.patch_infer import _IMAGENET_MEAN, _IMAGENET_STD
                            patch_rgb = (patch_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
                        x = torch.from_numpy(patch_rgb).permute(2,0,1).unsqueeze(0).to(device)
                        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
                        prob_acc[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += prob
                        weight[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]   += 1.0
            avg_prob = prob_acc / np.maximum(weight, 1.0)
            # Flatten to 1D for threshold sweep (whole-image level)
            all_probs_fold.append(avg_prob.flatten())
            all_targets_fold.append((gt > 0).astype(np.float32).flatten())

        sweep = run_threshold_sweep(all_probs_fold, all_targets_fold, out_dir,
                                     fold_k=fold_k + 10)  # offset to avoid name clash with non-patch sweeps
        opt_thr = sweep["optimal_threshold"]
        print(f"\n  [Fold {fold_k+1}] Opt thr = {opt_thr:.2f}  "
              f"Dice@opt = {sweep['best_dice']:.4f}")

        _, _, pred_c, gt_c, stms = _validate_full_images(
            model, device, val_paths, masks_dir, imagenet_norm, thr=opt_thr
        )
        all_pred_covers.extend(pred_c)
        all_gt_covers.extend(gt_c)
        all_stems.extend(stms)

        fold_results.append({
            "fold": fold_k,
            "val_stems":          [p.stem for p in val_paths],
            "best_val_dice":      round(train_res["best_val_dice"], 4),
            "best_val_iou":       round(train_res["best_val_iou"],  4),
            "optimal_threshold":  opt_thr,
            "sweep_dice_at_opt":  round(sweep["best_dice"], 4),
        })

    dice_vals = [r["best_val_dice"] for r in fold_results]
    iou_vals  = [r["best_val_iou"]  for r in fold_results]
    thr_vals  = [r["optimal_threshold"] for r in fold_results]

    print(f"\n{'═'*55}")
    print(f"  PATCH CV SUMMARY ({MODEL_TYPE}  loss={LOSS_TYPE})")
    print(f"{'─'*55}")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}:  Dice={r['best_val_dice']:.4f}  "
              f"IoU={r['best_val_iou']:.4f}  opt_thr={r['optimal_threshold']:.2f}")
    print(f"{'─'*55}")
    print(f"  Mean Dice = {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    print(f"  Mean IoU  = {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
    print(f"  Mean opt thr = {np.mean(thr_vals):.3f}")
    print(f"{'═'*55}\n")

    cover_result = save_cover_analysis(all_pred_covers, all_gt_covers, all_stems, out_dir)

    suffix = out_name or f"patch_{LOSS_TYPE}"
    result = {
        "model_type": MODEL_TYPE, "loss_type": LOSS_TYPE,
        "patch_size": PATCH_SIZE, "overlap": OVERLAP,
        "patch_epochs": PATCH_EPOCHS, "n_folds": N_FOLDS,
        "mean_val_dice":  round(float(np.mean(dice_vals)), 4),
        "std_val_dice":   round(float(np.std(dice_vals)),  4),
        "mean_val_iou":   round(float(np.mean(iou_vals)),  4),
        "std_val_iou":    round(float(np.std(iou_vals)),   4),
        "mean_optimal_threshold": round(float(np.mean(thr_vals)), 3),
        "per_fold": fold_results,
        "cover_correlation": cover_result,
    }
    json_path = out_dir / f"cv_results_{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch-based training")
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold stratified CV (stratified by image)")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    images_dir = ROOT / "data" / "raw_for_annot"  / BATCH_NAME
    masks_dir  = ROOT / "data" / "processed_annot" / BATCH_NAME / "bryozoan_gt_masks"

    if not masks_dir.exists():
        print("[INFO] GT masks folder missing."); return

    if args.cv:
        run_cv_patch(images_dir, masks_dir, out_name=args.out_name)
    else:
        print("Use --cv to run cross-validation.")


if __name__ == "__main__":
    main()
