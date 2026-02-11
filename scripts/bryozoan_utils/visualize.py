import cv2
import numpy as np
from pathlib import Path

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Red overlay for mask pixels.
    """
    out = img_bgr.copy()
    m = mask.copy()
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    red = np.zeros_like(out)
    red[:, :, 2] = 255

    mask3 = cv2.merge([m, m, m])
    out = np.where(mask3 > 0, (alpha * red + (1 - alpha) * out).astype(np.uint8), out)
    return out

def save_bryozoan_overlay(
    img_bgr: np.ndarray,
    blade_mask: np.ndarray,
    dense_mask: np.ndarray,
    edge_mask: np.ndarray,
    final_mask: np.ndarray,
    cover_pct: float,
    out_path: Path
) -> None:
    out = img_bgr.copy()

    # Add overlays in layers for debugging
    out = overlay_mask(out, dense_mask, alpha=0.35)
    out = overlay_mask(out, edge_mask, alpha=0.55)
    out = overlay_mask(out, final_mask, alpha=0.25)

    # Put text
    text = f"Bry cover: {cover_pct:.2f}%"
    cv2.putText(out, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
