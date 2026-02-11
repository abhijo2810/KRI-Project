import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from bryozoan_utils.edge_band import make_edge_band
from bryozoan_utils.dense_mat import detect_dense_bryozoans
from bryozoan_utils.edge_texture import detect_edge_bryozoans
from bryozoan_utils.fuse_masks import fuse_bryozoan_masks
from bryozoan_utils.visualize import save_bryozoan_overlay
from bryozoan_utils.cleanup import remove_small_components
from ml.bryozoan_predictor import predict_bryozoans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_NAME = "bryozoan_batch_01"

RAW_DIR = PROJECT_ROOT / "data" / "raw_for_annot" / BATCH_NAME
BLADE_MASK_DIR = PROJECT_ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"

OUT_MASKS_DIR = PROJECT_ROOT / "data" / "processed_annot" / BATCH_NAME / "bryozoan_masks"
OUT_OVERLAYS_DIR = PROJECT_ROOT / "outputs" / "overlays_bry_fast"
OUT_CSV = PROJECT_ROOT / "outputs" / "csv" / "bryozoan_fast_metrics.csv"

EDGE_BAND_WIDTH_PX = 35
WEIGHTS_PATH = PROJECT_ROOT / "models" / "bry_unet.pt"

def load_blade_mask(stem: str) -> np.ndarray:
    """
    Expects: data/processed_annot/<batch>/blade_masks/<stem>_blade_mask.png
    """
    p = BLADE_MASK_DIR / f"{stem}_blade_mask.png"
    if not p.exists():
        raise FileNotFoundError(f"Blade mask not found: {p}")

    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read blade mask: {p}")

    # Normalize to 0/255
    m = (m > 0).astype(np.uint8) * 255
    return m

def compute_cover_pct(final_mask: np.ndarray, blade_mask: np.ndarray) -> float:
    bry = (final_mask > 0).sum()
    blade = (blade_mask > 0).sum()
    if blade == 0:
        return 0.0
    return 100.0 * (bry / blade)

def main():
    OUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in RAW_DIR.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    if not image_paths:
        print(f"No images found in {RAW_DIR}")
        return

    rows = []

    for img_path in image_paths:
        stem = img_path.stem
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        try:
            blade_mask = load_blade_mask(stem)
        except Exception as e:
            print(f"Skipping (no blade mask): {stem} | {e}")
            continue

        band = make_edge_band(blade_mask, width_px=EDGE_BAND_WIDTH_PX)

        dense = detect_dense_bryozoans(img, blade_mask)
        edge = detect_edge_bryozoans(img, blade_mask, band)

        # Heuristic baseline
        heuristic_final = fuse_bryozoan_masks(dense, edge)

        # (optional) if you have remove_small_components utility, keep using it
        # heuristic_final = remove_small_components(heuristic_final, min_area=120)

        # Choose ML if weights exist; otherwise fallback to heuristic
        final_mask, mode, note = predict_bryozoans(
            img_bgr=img,
            blade_mask=blade_mask,
            heuristic_mask=heuristic_final,
            weights_path=WEIGHTS_PATH
        )

        cover_pct = compute_cover_pct(final_mask, blade_mask)

        # Save masks
        cv2.imwrite(str(OUT_MASKS_DIR / f"{stem}_dense.png"), dense)
        cv2.imwrite(str(OUT_MASKS_DIR / f"{stem}_edge.png"), edge)
        cv2.imwrite(str(OUT_MASKS_DIR / f"{stem}_final.png"), final_mask)

        # Save overlay for meeting
        save_bryozoan_overlay(
            img_bgr=img,
            blade_mask=blade_mask,
            dense_mask=dense,
            edge_mask=edge,
            final_mask=final_mask,
            cover_pct=cover_pct,
            out_path=OUT_OVERLAYS_DIR / f"{stem}_overlay.png"
        )

        rows.append({
            "image": stem,
            "bry_cover_percent": cover_pct,
            "edge_band_width_px": EDGE_BAND_WIDTH_PX,
            "detector_mode": mode,
            "detector_note": note,
        })

        print(f"[OK] {stem}: {cover_pct:.2f}% | mode={mode} ({note})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved metrics: {OUT_CSV}")
    print(f"Saved overlays: {OUT_OVERLAYS_DIR}")

if __name__ == "__main__":
    main()




