# scripts/pipeline_utils.py
"""Shared helpers used by multiple pipeline scripts."""

import json
from pathlib import Path

import cv2
import numpy as np


def load_blade_mask(mask_dir: Path, stem: str) -> np.ndarray:
    """Load a saved blade mask as a binary uint8 array (0/255)."""
    p = mask_dir / f"{stem}_blade_mask.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Missing blade mask: {p}")
    return (m > 0).astype(np.uint8) * 255


def load_pixels_per_mm(calib_path: Path, fallback: float = 4.0) -> float:
    """
    Load pixels-per-mm from calibration.json.

    IMPORTANT: calibration.json must contain an actual measured value.
    The fallback (4.0) is a placeholder — all physical measurements
    (area_cm2, invasion_depth_mm) will be wrong if it is used.
    """
    try:
        if calib_path.exists() and calib_path.stat().st_size > 0:
            with open(calib_path) as f:
                data = json.load(f)
            val = data.get("default", {}).get("pixels_per_mm", None)
            if val is not None:
                return float(val)
    except Exception as e:
        print(f"[warn] Could not parse {calib_path}: {e}")
    print(
        f"[warn] Using fallback pixels_per_mm={fallback}. "
        "Measure from a ruler in your images and update data/calibration.json."
    )
    return float(fallback)
