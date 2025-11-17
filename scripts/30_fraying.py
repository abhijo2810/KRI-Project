from pathlib import Path
import cv2, math, numpy as np, pandas as pd
from skimage import measure, morphology

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
IN_DIR = ROOT / "data" / "raw"
CSV_DIR = ROOT / "outputs" / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])

# ---------- Reuse segmentation ----------
def segment_blade(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    labels = measure.label(th > 0, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(th, dtype=np.uint8)
    props = measure.regionprops(labels)
    largest = max(props, key=lambda r: r.area)
    mask = (labels == largest.label).astype(np.uint8) * 255
    mask = morphology.convex_hull_image(mask > 0).astype(np.uint8) * 255
    return mask

# ---------- Metrics ----------
def roughness(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.nan
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    if A <= 0: return np.nan
    return (P / (2.0 * math.sqrt(math.pi * A))) - 1.0

def rim_edge_density(img_bgr, mask, rim_px=8):
    dil = cv2.dilate(mask, np.ones((rim_px, rim_px), np.uint8))
    ero = cv2.erode(mask, np.ones((rim_px, rim_px), np.uint8))
    band = cv2.subtract(dil, ero) > 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    total = band.sum()
    return (edges & band).sum() / total if total > 0 else np.nan

def micro_holes(mask, min_area=20, max_area=1000):
    filled = morphology.remove_small_holes(mask > 0, area_threshold=max_area*2)
    holes = (filled.astype(np.uint8)*255) - mask
    labels = measure.label(holes > 0)
    areas = [r.area for r in measure.regionprops(labels)]
    return sum(1 for a in areas if min_area <= a <= max_area)

# ---------- Main loop ----------
rows = []
for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[warn] unreadable: {p.name}")
        continue
    mask = segment_blade(img)
    rows.append({
        "image": p.name,
        "roughness": roughness(mask),
        "rim_edge_density": rim_edge_density(img, mask, rim_px=8),
        "micro_holes": micro_holes(mask)
    })

df = pd.DataFrame(rows)
out_csv = CSV_DIR / "fraying.csv"
df.to_csv(out_csv, index=False)
print(f"Processed {len(rows)} images")
print("Wrote:", out_csv)