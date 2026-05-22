# scripts/convert_goodnotes_annotations.py
"""
Converts GoodNotes red-highlighter annotations into binary bryozoan GT masks.

Usage:
  1. Put all exported GoodNotes JPGs into data/goodnotes_exports/.
  2. Create a mapping file (data/goodnotes_exports/mapping.txt) with lines like:
       Heavy_annot_1 -> 3.1
       Med_annot_1   -> 5.2
       Light_annot_1 -> 1.3
       no3.1         -> no3.1
     Lines starting with # are comments.
  3. From the project root, run:
       python scripts/convert_goodnotes_annotations.py

Output masks go to:
  data/processed_annot/bryozoan_batch_01/bryozoan_gt_masks/

Each mask: white (255) = bryozoan, black (0) = background.
Next step: python -m ml.train_unet
"""

from pathlib import Path
import cv2
import numpy as np

# ---------- Paths ----------
ROOT       = Path(__file__).resolve().parents[1]
BATCH_NAME = "bryozoan_batch_01"

ANNOT_DIR    = ROOT / "data" / "goodnotes_exports"
MAPPING_FILE = ANNOT_DIR / "mapping.txt"
OUT_DIR      = ROOT / "data" / "processed_annot" / BATCH_NAME / "bryozoan_gt_masks"
RAW_DIR      = ROOT / "data" / "raw_for_annot" / BATCH_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Red detection config ----------
# GoodNotes red highlighter in HSV. Hue for red wraps around 0/180 in
# OpenCV, so we check two ranges. Adjust SAT_MIN/VAL_MIN for washed-out exports.
RED_RANGES = [
    ((0,   80, 80), (10,  255, 255)),
    ((165, 80, 80), (180, 255, 255)),
]

MIN_COMPONENT_AREA = 300
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def detect_red_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Extract a binary mask of red-highlighted regions. Returns uint8 {0,255}."""
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for (lo, hi) in RED_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)

    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255

    return clean


def load_mapping(mapping_file: Path) -> dict:
    """Parse mapping.txt. Format per line:  annotation_stem -> original_stem"""
    mapping = {}
    if not mapping_file.exists():
        print(f"[warn] No mapping file found at: {mapping_file}")
        print("       Create mapping.txt in data/goodnotes_exports/.")
        print("       Format:  Heavy_annot_1 -> 3.1")
        return mapping

    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                print(f"[warn] Skipping malformed mapping line: {line!r}")
                continue
            left, right = line.split("->", 1)
            mapping[left.strip()] = right.strip()

    return mapping


def original_size(orig_stem: str) -> tuple | None:
    """Return (width, height) of the matching raw image, or None if not found."""
    for ext in [".JPG", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        p = RAW_DIR / f"{orig_stem}{ext}"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
    return None


# ---------- Main ----------
def main():
    exts = {".jpg", ".jpeg", ".png", ".JPG"}
    annot_files = sorted([p for p in ANNOT_DIR.iterdir() if p.suffix in exts])

    if not annot_files:
        print(f"No annotation files found in: {ANNOT_DIR}")
        print("Export your GoodNotes pages as JPGs into that folder and rerun.")
        return

    mapping = load_mapping(MAPPING_FILE)
    print(f"Found {len(annot_files)} annotation files, {len(mapping)} mapping entries.\n")

    success = 0
    skipped = 0

    for annot_path in annot_files:
        stem      = annot_path.stem
        orig_stem = mapping.get(stem)
        if orig_stem is None:
            print(f"[skip] No mapping for: {stem}  (add it to mapping.txt)")
            skipped += 1
            continue

        img = cv2.imread(str(annot_path))
        if img is None:
            print(f"[warn] Could not read: {annot_path.name}")
            skipped += 1
            continue

        mask = detect_red_mask(img)

        size = original_size(orig_stem)
        if size is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        else:
            print(f"[warn] Original image not found for stem '{orig_stem}', "
                  "mask kept at annotation resolution.")

        out_path = OUT_DIR / f"{orig_stem}_bry_gt.png"
        cv2.imwrite(str(out_path), mask)

        coverage = 100.0 * (mask > 0).sum() / mask.size
        print(f"[OK] {stem:<20} -> {orig_stem}_bry_gt.png  "
              f"({coverage:.1f}% pixels marked)")
        success += 1

    print(f"\nDone. {success} masks written to: {OUT_DIR}")
    if skipped:
        print(f"      {skipped} files skipped (check mapping.txt).")


if __name__ == "__main__":
    main()
