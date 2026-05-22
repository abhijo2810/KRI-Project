# scripts/35_invasion_depth.py
"""
Geometric invasion depth analysis.

For each blade image, casts inward rays from the blade boundary toward the
blade centroid. Detects where saturation drops (bryozoan signature: gray/white
colonies have lower HSV saturation than clean kelp). Reports how far the
invasion front extends and what fraction of the perimeter is affected.

Outputs per image:
    invasion_extent_pct    -- % of perimeter rays showing detected invasion
    invasion_depth_mean_mm -- mean distance from edge to invasion front (mm)
    invasion_depth_max_mm  -- deepest invasion detected (mm)
    n_rays_total           -- number of perimeter rays sampled
    n_rays_invaded         -- number of those rays with detected invasion

Overlay (*_inv.png): blade contour in green, invasion boundary coloured
yellow (shallow) to red (deep).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
from pipeline_utils import load_blade_mask, load_pixels_per_mm

# ---------- Paths ----------
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
BATCH_NAME  = "bryozoan_batch_01"

IN_DIR      = ROOT / "data" / "raw_for_annot" / BATCH_NAME
MASK_DIR    = ROOT / "data" / "processed_annot" / BATCH_NAME / "blade_masks"
OVERLAY_DIR = ROOT / "outputs" / "overlays"
CSV_DIR     = ROOT / "outputs" / "csv"
CALIB_PATH  = ROOT / "data" / "calibration.json"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

PIXELS_PER_MM = load_pixels_per_mm(CALIB_PATH)

# ---------- Algorithm config ----------
# At 4 px/mm, MAX_DEPTH_PX=120 covers 30 mm — enough for most colonies.
MAX_DEPTH_PX  = 120
RAY_STEP_PX   = 2
CONTOUR_STEP  = 15  # sample every Nth contour point

# Saturation must drop to this fraction of the clean-kelp reference
# to count as invaded. 0.55 = 55% of reference S value.
SAT_DROP_RATIO = 0.55

MIN_RAY_SAMPLES = 8


# ---------- Core geometry ----------
def _blade_centroid(mask: np.ndarray) -> np.ndarray:
    M = cv2.moments(mask)
    if M["m00"] == 0:
        h, w = mask.shape
        return np.array([w / 2.0, h / 2.0])
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


def _inward_unit(pt: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    d = centroid - pt
    n = np.linalg.norm(d)
    return d / n if n > 1e-8 else np.array([0.0, 1.0])


def _sample_ray(
    hsv: np.ndarray,
    blade_mask: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
) -> tuple:
    """Walk inward along a ray, collecting HSV saturation values."""
    h, w = blade_mask.shape
    depths, s_vals = [], []

    for d in range(RAY_STEP_PX, MAX_DEPTH_PX + RAY_STEP_PX, RAY_STEP_PX):
        pt = start + direction * d
        x, y = int(round(pt[0])), int(round(pt[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if blade_mask[y, x] == 0:
            break
        depths.append(d)
        s_vals.append(float(hsv[y, x, 1]))

    return np.array(depths), np.array(s_vals)


def _detect_invasion_depth(depths: np.ndarray, s_vals: np.ndarray) -> float | None:
    """
    Find the depth (px from edge) where bryozoan invasion begins.

    Reference: median saturation of the innermost 25% of the ray
    (assumed clean kelp interior). Invasion is flagged where saturation
    drops to SAT_DROP_RATIO * reference.
    """
    if len(depths) < MIN_RAY_SAMPLES:
        return None

    ref_n = max(2, len(s_vals) // 4)
    ref_s = float(np.median(s_vals[-ref_n:]))

    if ref_s < 10:  # bleached kelp — no meaningful baseline
        return None

    threshold = ref_s * SAT_DROP_RATIO

    # 3-sample moving average (valid mode avoids boundary zero-padding artifacts)
    if len(s_vals) >= 5:
        k = np.ones(3) / 3.0
        pad        = len(k) // 2
        s_padded   = np.pad(s_vals, pad, mode="edge")
        s_smooth   = np.convolve(s_padded, k, mode="valid")
    else:
        s_smooth = s_vals

    invaded = depths[s_smooth < threshold]
    return float(invaded.max()) if len(invaded) > 0 else None


# ---------- Per-image analysis ----------
def _analyze(img_bgr: np.ndarray, blade_mask: np.ndarray) -> tuple:
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    centroid = _blade_centroid(blade_mask)

    cnts, _ = cv2.findContours(blade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, []

    contour = max(cnts, key=cv2.contourArea)
    n_pts   = len(contour)

    invasion_depths = []
    inv_pts         = []

    for i in range(0, n_pts, CONTOUR_STEP):
        pt        = contour[i][0].astype(float)
        direction = _inward_unit(pt, centroid)

        depths, s_vals = _sample_ray(hsv, blade_mask, pt, direction)
        depth = _detect_invasion_depth(depths, s_vals)

        if depth is not None:
            invasion_depths.append(depth)
            bx = int(round(pt[0] + direction[0] * depth))
            by = int(round(pt[1] + direction[1] * depth))
            inv_pts.append((bx, by, depth))

    # Correct denominator: actual number of rays cast, not floor division
    n_sampled  = len(range(0, n_pts, CONTOUR_STEP))
    extent_pct = 100.0 * len(invasion_depths) / max(n_sampled, 1)

    if invasion_depths:
        mean_mm = float(np.mean(invasion_depths)) / PIXELS_PER_MM
        max_mm  = float(np.max(invasion_depths))  / PIXELS_PER_MM
    else:
        mean_mm = max_mm = 0.0

    metrics = {
        "invasion_extent_pct":    round(extent_pct, 2),
        "invasion_depth_mean_mm": round(mean_mm, 3),
        "invasion_depth_max_mm":  round(max_mm, 3),
        "n_rays_total":           n_sampled,
        "n_rays_invaded":         len(invasion_depths),
    }
    return metrics, inv_pts


def _save_overlay(img_bgr, blade_mask, inv_pts, out_path):
    overlay = img_bgr.copy()

    cnts, _ = cv2.findContours(blade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 220, 0), 2)

    if inv_pts:
        d_max = max(d for (_, _, d) in inv_pts) or 1.0
        for (x, y, d) in inv_pts:
            ratio = d / d_max
            g = int(220 * (1.0 - ratio))
            cv2.circle(overlay, (x, y), 4, (0, g, 220), -1)

    cv2.imwrite(str(out_path), overlay)


# ---------- Main ----------
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS])
rows  = []

print(f"pixels_per_mm used: {PIXELS_PER_MM}")
print(f"Found {len(paths)} images\n")

for img_path in paths:
    stem = img_path.stem
    img  = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] unreadable: {img_path.name}")
        continue

    try:
        blade_mask = load_blade_mask(MASK_DIR, stem)
    except FileNotFoundError:
        print(f"[warn] no blade mask, skipping: {stem}")
        continue

    metrics, inv_pts = _analyze(img, blade_mask)
    if metrics is None:
        print(f"[warn] no contour found: {stem}")
        continue

    _save_overlay(img, blade_mask, inv_pts, OVERLAY_DIR / f"{stem}_inv.png")

    rows.append({"image": stem, **metrics})
    print(
        f"[OK] {stem:<12}  extent={metrics['invasion_extent_pct']:5.1f}%  "
        f"mean={metrics['invasion_depth_mean_mm']:.2f}mm  "
        f"max={metrics['invasion_depth_max_mm']:.2f}mm"
    )

df      = pd.DataFrame(rows)
out_csv = CSV_DIR / "invasion_depth.csv"
df.to_csv(out_csv, index=False)
print(f"\nWrote: {out_csv}")
print(f"Overlays in: {OVERLAY_DIR}  (*_inv.png)")
