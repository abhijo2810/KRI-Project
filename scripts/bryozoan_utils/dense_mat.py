import cv2
import numpy as np

def detect_dense_bryozoans(
    img_bgr: np.ndarray,
    blade_mask: np.ndarray,
    dark_pct: float = 25.0,
    low_sat_pct: float = 35.0,
):
    """
    Adaptive dense bryozoan detector (Option B).

    Logic:
    - Restrict to blade region only
    - Convert to HSV
    - Compute per-image percentiles of V (brightness) and S (saturation)
    - Dense bryozoans = pixels that are BOTH:
        - darker than the blade's dark_pct percentile
        - less saturated than the blade's low_sat_pct percentile

    Returns:
        uint8 mask {0,255}
    """

    # Ensure blade mask is binary 0/255
    m = blade_mask.copy()
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    blade = m > 0
    if blade.sum() == 0:
        return np.zeros_like(m)

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Only consider blade pixels
    V_vals = V[blade]
    S_vals = S[blade]

    # Adaptive thresholds (percentiles)
    V_THR = np.percentile(V_vals, dark_pct)
    S_THR = np.percentile(S_vals, low_sat_pct)

    # Dark + low saturation
    dark = (V <= V_THR)
    low_sat = (S <= S_THR)

    dense = (dark & low_sat & blade).astype(np.uint8) * 255

    # Morphological cleanup (connect colonies, remove pepper noise)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, k, iterations=1)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, k, iterations=2)

    return dense