# scripts/calibrate_ruler.py
"""
Automatic pixels-per-mm calibration from the ruler visible in each image.

How it works:
  1. Auto-locates the horizontal white ruler strip in each image.
  2. Extracts a 1-D brightness profile across the ruler.
  3. Finds tick-mark positions as local brightness minima using peak detection.
  4. Computes pixels_per_mm from the median inter-tick spacing (each tick = 1 mm).
  5. Rejects per-image estimates where too few ticks were found
     (indicates the mm ticks were not resolved — likely wrong region).
  6. Writes the batch median to data/calibration.json.
  7. Saves a QA diagnostic strip image to outputs/overlays/calibration_qa.png.

Usage:
    python scripts/calibrate_ruler.py

Manual fallback (if auto-detection fails for your images):
  See the printed instructions at the end of this script, or the
  MANUAL CALIBRATION section in the project README.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent

BATCH_NAME = "bryozoan_batch_01"
RAW_DIR    = ROOT / "data" / "raw_for_annot" / BATCH_NAME
CALIB_PATH = ROOT / "data" / "calibration.json"
QA_DIR     = ROOT / "outputs" / "overlays"
QA_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".JPG", ".png", ".tif", ".tiff"}

# ── Tuning constants ──────────────────────────────────────────────────────────
# Only trust a measurement if we found at least this many ticks.
# Each tick is 1 mm, so this is the minimum ruler length in pixels / px_per_mm.
# At ~7 px/mm a 30 cm ruler gives ~300 ticks; require at least 350 for safety.
MIN_TICKS = 350

# Peak detection: minimum height as a fraction of the inverted-profile maximum.
PEAK_HEIGHT_FRAC = 0.20

# Minimum pixel distance between adjacent detected ticks (avoids double-peaks).
MIN_TICK_DIST_PX = 4
# ─────────────────────────────────────────────────────────────────────────────


def _find_ruler_strip(gray: np.ndarray) -> tuple[int, int] | None:
    """
    Return (row_top, row_bot) of the horizontal ruler strip.
    Searches for the first wide bright band (mean > 80 across the central half).
    Returns None if no ruler is found.
    """
    h, w = gray.shape
    cx0, cx1 = w // 4, 3 * w // 4

    ruler_top = None
    for row in range(30, h // 2):
        if gray[row, cx0:cx1].mean() > 80:
            ruler_top = row
            break
    if ruler_top is None:
        return None

    # Extend downward while still reasonably bright
    ruler_bot = ruler_top + 15
    for row in range(ruler_top + 10, min(ruler_top + 150, h)):
        if gray[row, cx0:cx1].mean() > 100:
            ruler_bot = row

    return ruler_top, ruler_bot + 15


def _measure_ppm(img_path: Path) -> tuple[float | None, str, np.ndarray | None]:
    """
    Measure pixels_per_mm for one image.

    Returns:
        ppm   -- float, or None on failure
        info  -- human-readable status string
        strip -- BGR crop of the detected ruler strip (for QA), or None
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "could not read file", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bounds = _find_ruler_strip(gray)
    if bounds is None:
        return None, "ruler strip not found", None

    row_top, row_bot = bounds
    strip_bgr = img[row_top:row_bot, :]
    strip_gray = gray[row_top:row_bot, :]
    profile = strip_gray.mean(axis=0)

    # Horizontal extent of the white ruler
    bright_cols = np.where(profile > 80)[0]
    if len(bright_cols) < 200:
        return None, "ruler too short in profile", strip_bgr

    x0, x1 = int(bright_cols[0]), int(bright_cols[-1])
    ruler_profile = profile[x0:x1]

    # Ticks are dark on white → invert, then find peaks
    inv = ruler_profile.max() - ruler_profile
    if inv.max() < 8:
        return None, "no contrast in ruler strip", strip_bgr

    peaks, _ = find_peaks(
        inv,
        height=inv.max() * PEAK_HEIGHT_FRAC,
        distance=MIN_TICK_DIST_PX,
    )

    if len(peaks) < MIN_TICKS:
        return (
            None,
            f"only {len(peaks)} ticks detected (< {MIN_TICKS} required; "
            "possibly detected cm marks only — check QA image)",
            strip_bgr,
        )

    span_px = int(peaks[-1]) - int(peaks[0])
    ppm = span_px / (len(peaks) - 1)
    return ppm, f"{len(peaks)} ticks  rows {row_top}–{row_bot}", strip_bgr


def _save_qa_image(qa_strips: list[tuple[str, np.ndarray, float | None]]) -> None:
    """Save a contact sheet of ruler strips with measured px/mm annotated."""
    h_target = 80
    panels = []
    for name, strip, ppm in qa_strips[:20]:  # cap at 20 images
        scale = h_target / strip.shape[0]
        w_new = int(strip.shape[1] * scale)
        panel = cv2.resize(strip, (min(w_new, 600), h_target))
        label = f"{name}: {ppm:.2f}" if ppm else f"{name}: FAILED"
        cv2.putText(panel, label, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)

    # Pad all panels to the same width
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        pad = max_w - p.shape[1]
        padded.append(cv2.copyMakeBorder(p, 0, 0, 0, pad, cv2.BORDER_CONSTANT))

    qa_img = cv2.vconcat(padded)
    out = QA_DIR / "calibration_qa.png"
    cv2.imwrite(str(out), qa_img)
    print(f"QA image saved to: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
image_paths = sorted([p for p in RAW_DIR.iterdir() if p.suffix in EXTS])
if not image_paths:
    print(f"[ERROR] No images found in {RAW_DIR}")
    sys.exit(1)

print(f"Measuring ruler in {len(image_paths)} images...\n")

good_ppm = []
qa_strips = []

for img_path in image_paths:
    ppm, info, strip = _measure_ppm(img_path)
    status = f"{ppm:.4f} px/mm" if ppm else "FAILED"
    print(f"  {img_path.name:<20s}  {status:<18s}  {info}")
    if strip is not None:
        qa_strips.append((img_path.name, strip, ppm))
    if ppm is not None:
        good_ppm.append(ppm)

print(f"\n{'─'*55}")
print(f"Successful measurements: {len(good_ppm)} / {len(image_paths)}")

if not good_ppm:
    print("\n[ERROR] No valid measurements. Check QA image and use manual calibration.")
    _save_qa_image(qa_strips)
    sys.exit(1)

median_ppm = float(np.median(good_ppm))
mean_ppm   = float(np.mean(good_ppm))
std_ppm    = float(np.std(good_ppm))

print(f"\nMedian pixels_per_mm : {median_ppm:.4f}")
print(f"Mean   pixels_per_mm : {mean_ppm:.4f}")
print(f"Std                  : {std_ppm:.4f}  ({100*std_ppm/median_ppm:.1f}% CV)")

if std_ppm / median_ppm > 0.10:
    print(
        "\n[WARN] CV > 10% — images in this batch may have been shot at different "
        "camera distances.\n"
        "       Consider splitting into sub-batches with separate calibration values,\n"
        "       or use per-image calibration by checking the printed values above."
    )

# ── Write calibration.json ────────────────────────────────────────────────────
calib = {"default": {"pixels_per_mm": round(median_ppm, 4)}}
with open(CALIB_PATH, "w") as f:
    json.dump(calib, f, indent=2)

print(f"\nWritten to: {CALIB_PATH}")
print(json.dumps(calib, indent=2))

_save_qa_image(qa_strips)

# ── Manual calibration instructions ──────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MANUAL CALIBRATION (if auto-detection gave bad results)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Step 1. Open any image in an app that shows pixel coordinates.
         • Mac Preview: Tools → Show Inspector (⌘I) shows x/y as you hover.
         • FIJI/ImageJ: the status bar shows pixel coordinates live.
         • Photoshop: Window → Info panel.

 Step 2. Hover over the "0 cm" mark on the ruler.
         Note the x coordinate (call it x0).

 Step 3. Hover over a mark 10 cm (= 100 mm) further along the ruler.
         Note the x coordinate (call it x1).
         Use a wide gap (10 cm minimum) to reduce click error.

 Step 4. Calculate:
           pixels_per_mm = (x1 - x0) / 100.0

 Step 5. Open  data/calibration.json  and set:
           {"default": {"pixels_per_mm": <your value>}}

 Step 6. Re-run scripts 20, 35, and 50 to recompute physical measurements.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
