# scripts/00_check_images.py
from pathlib import Path
import cv2, numpy as np

# Resolve paths relative to this script, not the shell's CWD
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IN_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "overlays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])

print(f"Project root: {PROJECT_ROOT}")
print(f"Looking in:   {IN_DIR}")
print(f"Found {len(paths)} files in data/raw")

bad = []
dims = []

for p in paths:
    img = cv2.imread(str(p))  # BGR
    if img is None:
        bad.append(p.name)
        continue
    h, w = img.shape[:2]
    dims.append((w, h))

print(f"Readable: {len(paths)-len(bad)} | Unreadable: {len(bad)}")
if bad:
    print("Unreadable (likely unsupported format):")
    for b in bad[:20]:
        print(" -", b)

# Contact sheet of first 12
thumbs = []
for p in paths[:12]:
    img = cv2.imread(str(p))
    if img is None: continue
    scale = 240.0 / img.shape[1]
    th = cv2.resize(img, (240, int(img.shape[0]*scale)))
    thumbs.append(th)

if thumbs:
    rows = []
    row = []
    for i, th in enumerate(thumbs, 1):
        row.append(th)
        if i % 3 == 0:
            rows.append(cv2.hconcat(row)); row = []
    if row:  # pad last row
        h = row[0].shape[0]
        while len(row) < 3:
            row.append(np.zeros((h, 240, 3), dtype=np.uint8))
        rows.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows)
    out_path = OUT_DIR / "contact_sheet_step1.png"
    cv2.imwrite(str(out_path), grid)
    print("Wrote:", out_path)

if dims:
    ws, hs = zip(*dims)
    print(f"Min size: {min(ws)}x{min(hs)} | Max size: {max(ws)}x{max(hs)}")