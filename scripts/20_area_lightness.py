# scripts/20_area_lightness.py
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_utils import load_blade_mask, load_pixels_per_mm

# ---------- Paths ----------
ROOT       = SCRIPT_DIR.parent
BATCH_NAME = "bryozoan_batch_01"
IN_DIR     = ROOT / "data" / "raw_for_annot" / BATCH_NAME
MASK_DIR   = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"

OVERLAY_DIR = ROOT / "outputs" / "overlays"
CSV_DIR     = ROOT / "outputs" / "csv"
CALIB_PATH  = ROOT / "data" / "calibration.json"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])

PIXELS_PER_MM = load_pixels_per_mm(CALIB_PATH, fallback=4.0)
print("pixels_per_mm used:", PIXELS_PER_MM)


# ---------- Metrics ----------
def px_area_to_cm2(area_px: float, px_per_mm: float) -> float:
    """Convert pixel area to square centimeters using calibration."""
    if area_px <= 0:
        return float("nan")
    mm2 = area_px / (px_per_mm ** 2)
    return mm2 / 100.0  # 100 mm² = 1 cm²


def lab_stats(img_bgr: np.ndarray, mask: np.ndarray) -> tuple:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    m = mask > 0
    if m.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    Lvals = L[m]
    L_mean   = float(Lvals.mean())
    L_std    = float(Lvals.std())
    pct_dark = float((Lvals < 100).sum()) / float(Lvals.size) * 100.0
    return L_mean, L_std, pct_dark


# ---------- Main loop ----------
rows = []
for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[warn] unreadable: {p.name}")
        continue

    try:
        mask = load_blade_mask(MASK_DIR, p.stem)
    except FileNotFoundError as e:
        print(f"[warn] skipping {p.name}: {e}")
        continue

    area_px  = int((mask > 0).sum())
    area_cm2 = px_area_to_cm2(area_px, PIXELS_PER_MM)

    L_mean, L_std, pct_dark = lab_stats(img, mask)

    overlay = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(str(OVERLAY_DIR / f"{p.stem}_seg.png"), overlay)

    rows.append({
        "image":    p.stem,
        "area_px":  area_px,
        "area_cm2": area_cm2,
        "L_mean":   L_mean,
        "L_std":    L_std,
        "pct_dark": pct_dark,
    })

df = pd.DataFrame(rows)
out_csv = CSV_DIR / "area_lightness.csv"
df.to_csv(out_csv, index=False)

print(f"Processed {len(rows)} images")
print("Wrote:", out_csv)
print("Overlays at:", OVERLAY_DIR)
