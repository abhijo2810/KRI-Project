# scripts/segment_blade_func.py
import cv2
import numpy as np

def crop_fixed(img, pad=80):
    """Remove ruler + table edges by cropping fixed pixels."""
    h, w = img.shape[:2]
    pad = min(pad, h // 4, w // 4)
    return img[pad:h-pad, pad:w-pad]

def segment_all_blades(img_bgr):
    """Segment all kelp blades using HSV color threshold + morphology."""

    # 1. Crop edges
    img = crop_fixed(img_bgr, pad=80)

    # 2. HSV segmentation for kelp amber/brown
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 40, 40])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 3. Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Fill holes
    mask_filled = mask.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)

    # 5. Keep only large components (blades)
    nb, output, stats, _ = cv2.connectedComponentsWithStats(mask_filled)
    final_mask = np.zeros_like(mask)

    min_size = 10000
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            final_mask[output == i] = 255

    return final_mask
