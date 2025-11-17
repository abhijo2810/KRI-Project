# scripts/20_area_lightness.py
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
IN_DIR = ROOT / "data" / "raw"
OVERLAY_DIR = ROOT / "outputs" / "overlays"
CSV_DIR = ROOT / "outputs" / "csv"
CALIB_PATH = ROOT / "data" / "calibration.json"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])

# ---------- Calibration ----------

def load_pixels_per_mm(calib_path, fallback=4.0):
    try:
        if calib_path.exists() and calib_path.stat().st_size > 0:
            with open(calib_path, "r") as f:
                data = json.load(f)
            return float(data.get("default", {}).get("pixels_per_mm", fallback))
        else:
            print(f"[warn] {calib_path} missing or empty. Using default {fallback}")
            return float(fallback)
    except Exception as e:
        print(f"[warn] Could not parse {calib_path}: {e}. Using default {fallback}")
        return float(fallback)

PIXELS_PER_MM = load_pixels_per_mm(CALIB_PATH, fallback=4.0)
print("pixels_per_mm used:", PIXELS_PER_MM)

# ---------- Segmentation ----------
def segment_blade(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    labels = measure.label(th > 0, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(th, dtype=np.uint8)
    props = measure.regionprops(labels)
    largest = max(props, key=lambda r: r.area)
    mask = (labels == largest.label).astype(np.uint8) * 255
    mask = morphology.convex_hull_image(mask > 0).astype(np.uint8) * 255
    return mask

def contour_and_area(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    return c, float(cv2.contourArea(c))

def px_area_to_cm2(area_px: float, px_per_mm: float) -> float:
    """Convert pixel area to square centimeters using calibration."""
    if area_px <= 0:
        return float("nan")
    mm_per_px = 1.0 / px_per_mm
    mm2 = area_px * (mm_per_px ** 2)
    return mm2 / 100.0  # 100 mm² = 1 cm²

# ---------- Lightness / discoloration ----------
def lab_stats(img_bgr, mask):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    m = mask > 0
    if m.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    Lvals = L[m]
    L_mean = float(Lvals.mean())
    L_std  = float(Lvals.std())
    # Simple discoloration proxy: % of pixels darker than L* < 100
    pct_dark = float((Lvals < 100).sum()) / float(Lvals.size) * 100.0
    return L_mean, L_std, pct_dark

# ---------- Main loop ----------
rows = []
for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[warn] unreadable: {p.name}")
        continue

    mask = segment_blade(img)
    cnt, area_px = contour_and_area(mask)
    area_cm2 = px_area_to_cm2(area_px, PIXELS_PER_MM)

    L_mean, L_std, pct_dark = lab_stats(img, mask)

    # Save overlay (green contour) for QA
    overlay = img.copy()
    if cnt is not None:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
    cv2.imwrite(str(OVERLAY_DIR / f"{p.stem}_seg.png"), overlay)

    rows.append({
        "image": p.name,
        "area_px": area_px,
        "area_cm2": area_cm2,
        "L_mean": L_mean,
        "L_std": L_std,
        "pct_dark": pct_dark
    })

df = pd.DataFrame(rows)
out_csv = CSV_DIR / "area_lightness.csv"
df.to_csv(out_csv, index=False)

print(f"pixels_per_mm used: {PIXELS_PER_MM}")
print(f"Processed {len(rows)} images")
print("Wrote:", out_csv)
print("Overlays at:", OVERLAY_DIR)