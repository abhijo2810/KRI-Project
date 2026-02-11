import cv2
import numpy as np

def remove_small_components(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """
    Remove connected components smaller than min_area pixels.
    mask: uint8 {0,255} or {0,1}
    returns: uint8 {0,255}
    """
    m = mask.copy()
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out
