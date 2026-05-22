# ml/bryozoan_predictor.py
import json
from pathlib import Path

import cv2
import numpy as np

# ImageNet stats (duplicated here to avoid importing the full factory at inference
# time when torch may not be available).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default inference threshold — overridden by optimal threshold from Task 3 once known.
DEFAULT_THR = 0.5


def _load_ckpt_config(weights_path: Path) -> dict:
    """
    Read the companion config JSON saved alongside the checkpoint.
    Falls back to custom_unet defaults for backward compatibility with
    checkpoints saved before the model factory was introduced.
    """
    cfg_path = weights_path.with_suffix(".json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {"model_type": "custom_unet", "input_size": 512, "imagenet_norm": False}


def _load_model(weights_path: Path):
    """Load and return (model, device, config). Returns (None, None, None) on failure."""
    try:
        import torch
        from ml.models.model_factory import build_model, select_device
    except Exception:
        return None, None, None

    if not weights_path.exists():
        return None, None, None

    cfg   = _load_ckpt_config(weights_path)
    model = build_model(cfg["model_type"])
    model, device = select_device(model)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device, cfg


def _infer(img_bgr: np.ndarray, model, device, cfg: dict, thr: float) -> np.ndarray:
    """Run one image through the model; return binary mask (0/255) at original resolution."""
    import torch

    size = cfg.get("input_size", 512)
    imagenet_norm = cfg.get("imagenet_norm", False)

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    if imagenet_norm:
        x = (x - _MEAN) / _STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    mask = (prob > thr).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def predict_bryozoans(
    img_bgr: np.ndarray,
    blade_mask: np.ndarray,
    heuristic_mask: np.ndarray,
    weights_path: Path,
    thr: float = DEFAULT_THR,
) -> tuple[np.ndarray, str, str]:
    """
    Returns (final_mask uint8 0/255, mode str, note str).

    Tries the trained ML model first; falls back to heuristic if weights are
    missing or the model cannot be loaded.
    """
    model, device, cfg = _load_model(weights_path)

    if model is not None:
        pred = _infer(img_bgr, model, device, cfg, thr)
        pred = cv2.bitwise_and(pred, (blade_mask > 0).astype(np.uint8) * 255)
        model_type = cfg.get("model_type", "unknown")
        return pred, "ml", f"{model_type} thr={thr}"

    return heuristic_mask, "heuristic", "no weights (fallback)"
