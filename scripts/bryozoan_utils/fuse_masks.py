import cv2
import numpy as np

def fuse_bryozoan_masks(dense_mask: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """
    Logical OR + cleanup.
    returns: uint8 {0,255} fused mask
    """
    d = dense_mask.astype(np.uint8)
    e = edge_mask.astype(np.uint8)

    fused = cv2.bitwise_or(d, e)

    # Optional cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k, iterations=1)

    return fused
