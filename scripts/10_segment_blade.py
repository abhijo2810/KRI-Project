# scripts/10_segment_blade.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

BATCH_NAME = "bryozoan_batch_01"

IN_DIR   = ROOT / "data" / "raw_for_annot" / BATCH_NAME
MASK_DIR = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"

OVERLAY_DIR = ROOT / "outputs" / "overlays"
CSV_DIR     = ROOT / "outputs" / "csv"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])


# ---------------------------------------------------------
# SEGMENTATION — detects ALL blades
# No crop: mask must align with the original image exactly.
# HSV bounds [5–35 hue, 40–255 sat, 40–255 val] target
# amber/brown kelp on a white background. Adjust if lighting
# or substrate colour changes between batches.
# ---------------------------------------------------------
def segment_all_blades(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([5,  40,  40])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Fill internal holes so each blade is a solid region
    mask_filled = mask.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)

    # Keep only large components (removes stray colour specks)
    min_size = 10_000
    nb, output, stats, _ = cv2.connectedComponentsWithStats(mask_filled)
    final_mask = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            final_mask[output == i] = 255

    return final_mask


def find_all_blade_contours(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
rows = []

for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[warn] could not read: {p.name}")
        continue

    mask     = segment_all_blades(img)
    cv2.imwrite(str(MASK_DIR / f"{p.stem}_blade_mask.png"), mask)
    contours = find_all_blade_contours(mask)

    overlay = img.copy()
    for c in contours:
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
    cv2.imwrite(str(OVERLAY_DIR / f"{p.stem}_seg.png"), overlay)

    total_area_px = int((mask > 0).sum())
    rows.append({"image": p.name, "area_px": total_area_px})

df = pd.DataFrame(rows)
df.to_csv(CSV_DIR / "area_pixels.csv", index=False)

print(f"Processed {len(rows)} images")
print("Overlays saved to:", OVERLAY_DIR)
print("CSV saved:", CSV_DIR / "area_pixels.csv")
