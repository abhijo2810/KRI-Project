import cv2
import numpy as np

def make_edge_band(blade_mask: np.ndarray, width_px: int = 20) -> np.ndarray:
    """
    Create an edge band (ring) around the blade boundary.

    blade_mask: uint8 {0,255} or {0,1}
    returns: uint8 {0,255} mask for edge band
    """
    m = blade_mask.copy()
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    k = max(1, int(width_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k + 1, 2*k + 1))

    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)

    band = cv2.subtract(dil, ero)
    # keep only within dilated blade region (optional safety)
    band = cv2.bitwise_and(band, dil)
    return band
