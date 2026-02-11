import cv2
import numpy as np

def detect_edge_bryozoans(
    img_bgr: np.ndarray,
    blade_mask: np.ndarray,
    edge_band: np.ndarray,
    grad_thresh: float = 18.0,
    sat_thresh: int = 25,
) -> np.ndarray:
    """
    Thin-edge bryozoan detector (fast + defensible):
    - only looks in edge_band
    - uses local contrast/texture (Sobel gradient magnitude)
    - uses HSV saturation gate to reduce shadow-only false positives

    returns: uint8 {0,255} mask
    """
    m = blade_mask.copy()
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    band = edge_band.copy()
    if band.max() == 1:
        band = (band * 255).astype(np.uint8)
    else:
        band = band.astype(np.uint8)

    # Only consider edge band within blade
    band = cv2.bitwise_and(band, m)

    # Gradient magnitude (texture / micro-edges)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Normalize to 0..255 for thresholding
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Saturation gate (helps reject pure shadows on kelp)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    # Candidates: high gradient AND enough saturation, only in band
    grad_mask = (mag_u8 >= int(grad_thresh)).astype(np.uint8) * 255
    sat_mask = (s >= int(sat_thresh)).astype(np.uint8) * 255

    edge = cv2.bitwise_and(grad_mask, sat_mask)
    edge = cv2.bitwise_and(edge, band)

    # Cleanup / connect thin structures
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k, iterations=1)

    return edge
