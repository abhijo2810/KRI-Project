# scripts/30_fraying.py
from pathlib import Path
import math
import sys
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_utils import load_blade_mask

# ---------- Paths ----------
ROOT       = SCRIPT_DIR.parent
BATCH_NAME = "bryozoan_batch_01"
IN_DIR     = ROOT / "data" / "raw_for_annot" / BATCH_NAME
MASK_DIR   = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"

CSV_DIR = ROOT / "outputs" / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])


# ---------- Metrics ----------
def roughness(mask: np.ndarray) -> float:
    """
    Excess-perimeter ratio: (P / (2 * sqrt(pi * A))) - 1.
    Zero for a perfect circle; larger values mean rougher boundary.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c)
    P = cv2.arcLength(c, True)
    if A <= 0:
        return np.nan
    return (P / (2.0 * math.sqrt(math.pi * A))) - 1.0


def rim_edge_density(img_bgr: np.ndarray, mask: np.ndarray, rim_px: int = 8) -> float:
    dil  = cv2.dilate(mask, np.ones((rim_px, rim_px), np.uint8))
    ero  = cv2.erode( mask, np.ones((rim_px, rim_px), np.uint8))
    band = cv2.subtract(dil, ero) > 0
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    total = band.sum()
    return float((edges & band).sum()) / total if total > 0 else np.nan


def micro_holes(mask: np.ndarray, min_area: int = 20, max_area: int = 1000) -> int:
    """Count internal holes with area in [min_area, max_area] pixels."""
    filled = morphology.remove_small_holes(mask > 0, area_threshold=max_area * 2)
    holes  = filled & ~(mask > 0)
    labels = measure.label(holes)
    areas  = [r.area for r in measure.regionprops(labels)]
    return sum(1 for a in areas if min_area <= a <= max_area)


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

    rows.append({
        "image":            p.stem,
        "roughness":        roughness(mask),
        "rim_edge_density": rim_edge_density(img, mask, rim_px=8),
        "micro_holes":      micro_holes(mask),
    })

df = pd.DataFrame(rows)
out_csv = CSV_DIR / "fraying.csv"
df.to_csv(out_csv, index=False)
print(f"Processed {len(rows)} images")
print("Wrote:", out_csv)
