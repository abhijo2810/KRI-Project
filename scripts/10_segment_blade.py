# scripts/10_segment_blade.py
from pathlib import Path
import cv2
import numpy as np
from skimage import measure
import pandas as pd

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
IN_DIR = ROOT / "data" / "raw"
OVERLAY_DIR = ROOT / "outputs" / "overlays"
CSV_DIR = ROOT / "outputs" / "csv"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Allowed extensions
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])


# ---------------------------------------------------------
# FIXED HARDCROP (More reliable than auto-crop)
# ---------------------------------------------------------
def crop_fixed(img, pad=80):
    """Remove ruler + table edges by cropping fixed pixels."""
    h, w = img.shape[:2]
    pad = min(pad, h // 4, w // 4)
    return img[pad:h-pad, pad:w-pad]


# ---------------------------------------------------------
# SEGMENTATION FUNCTION — Detects ALL blades
# ---------------------------------------------------------
def segment_all_blades(img_bgr):

    # 1. Hard-crop borders (stable)
    img = crop_fixed(img_bgr, pad=80)

    # 2. Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # These bounds work well for amber/brown kelp on white background
    lower = np.array([5, 40, 40])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # 3. Clean mask (NO BLUR - keep sharp edges)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Fill holes so contour is complete
    mask_filled = mask.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)

    # 5. Keep ALL large connected components (all blades)
    nb, output, stats, _ = cv2.connectedComponentsWithStats(mask_filled)
    final_mask = np.zeros_like(mask)

    min_size = 10000  # threshold to keep kelp pieces, remove junk

    for i in range(1, nb):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            final_mask[output == i] = 255

    return final_mask


# ---------------------------------------------------------
# FIND CONTOURS FOR ALL BLADES
# ---------------------------------------------------------
def find_all_blade_contours(mask):
    # External contours only, no approximation reduction
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
rows = []

for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARN] could not read: {p.name}")
        continue

    # segment blades
    mask = segment_all_blades(img)
    contours = find_all_blade_contours(mask)

    # draw outlines
    overlay = img.copy()
    for c in contours:
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)

    # save overlay
    out_name = OVERLAY_DIR / f"{p.stem}_seg.png"
    cv2.imwrite(str(out_name), overlay)

    # compute total area (optional)
    total_area_px = sum(cv2.contourArea(c) for c in contours)
    rows.append({"image": p.name, "area_px": total_area_px})

# save CSV
df = pd.DataFrame(rows)
df.to_csv(CSV_DIR / "area_pixels.csv", index=False)

print(f"Processed {len(rows)} images")
print("Overlays saved to:", OVERLAY_DIR)
print("CSV saved:", CSV_DIR / "area_pixels.csv")
