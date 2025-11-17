# scripts/40_bryozoan_unsup.py
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from segment_blade_func import segment_all_blades, crop_fixed

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
IN_DIR = ROOT / "data" / "raw"
OVERLAY_DIR = ROOT / "outputs" / "bryozoan_overlays"
CSV_DIR = ROOT / "outputs" / "csv"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LBP texture ----------
def lbp(gray, P=8, R=1):
    return local_binary_pattern(gray, P, R, method="default").astype(np.float32)

# ---------- Bryozoan detection ----------
def detect_bryozoans(img_bgr, mask_blade):
    m = mask_blade > 0
    if m.sum() < 50:
        return np.zeros_like(mask_blade)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0].astype(np.float32)
    a = lab[:,:,1].astype(np.float32)
    b = lab[:,:,2].astype(np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tex = lbp(gray)

    # Feature matrix inside blade
    X = np.stack([L[m], a[m], b[m], tex[m]], axis=1)

    # KMeans clustering
    km = KMeans(n_clusters=3, n_init=5, random_state=0)
    labels = km.fit_predict(X)

    # Choose cluster with darkest L → bryozoan
    L_mean = [X[labels==k,0].mean() for k in range(3)]
    bry_k = int(np.argmin(L_mean))

    bry = np.zeros_like(mask_blade)
    bry[m] = (labels == bry_k).astype(np.uint8) * 255

    return bry

# ---------- Main loop ----------
rows = []
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".bmp"}])

for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARN] unreadable: {p.name}")
        continue

    # 1. Crop FIRST (only crop once)
    img = crop_fixed(img, pad=80)

    # 2. Blade segmentation (NO more internal cropping)
    blade_mask = segment_all_blades(img)

    # Bryozoan mask inside blade
    bry_mask = detect_bryozoans(img, blade_mask)

    # Areas
    blade_area = int((blade_mask > 0).sum())
    bry_area = int((bry_mask > 0).sum())
    pct = (bry_area / blade_area * 100) if blade_area > 0 else 0

    # Overlay
    overlay = img.copy()
    overlay[bry_mask > 0] = (overlay[bry_mask > 0] * 0.4 + np.array([0,0,255]) * 0.6).astype("uint8")

    cnts,_ = cv2.findContours(blade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(overlay, [c], -1, (0,255,0), 2)

    out_file = OVERLAY_DIR / f"{p.stem}_bry.png"
    cv2.imwrite(str(out_file), overlay)

    rows.append({
        "image": p.name,
        "blade_area_px": blade_area,
        "bry_area_px": bry_area,
        "bry_percent": pct
    })

df = pd.DataFrame(rows)
df.to_csv(CSV_DIR / "bry_fast.csv", index=False)
print("Done. Saved:", CSV_DIR / "bry_fast.csv")



