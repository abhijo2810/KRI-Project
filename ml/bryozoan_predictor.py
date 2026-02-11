from pathlib import Path
import numpy as np
import cv2

def _try_load_torch_model(weights_path: Path):
    try:
        import torch
        from ml.models.unet import UNetSmall
    except Exception:
        return None, None

    if not weights_path.exists():
        return None, None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNetSmall().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def _infer_unet(img_bgr: np.ndarray, model, device, thr=0.5, size=512) -> np.ndarray:
    import torch
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    m = (prob > thr).astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m

def predict_bryozoans(img_bgr, blade_mask, heuristic_mask, weights_path: Path):
    """
    Returns:
      final_mask (uint8 0/255)
      mode: 'ml' or 'heuristic'
      note: string for logging
    """
    model, device = _try_load_torch_model(weights_path)

    if model is not None:
        pred = _infer_unet(img_bgr, model, device, thr=0.5, size=512)
        pred = cv2.bitwise_and(pred, (blade_mask > 0).astype(np.uint8) * 255)
        return pred, "ml", f"weights={weights_path.name}"

    return heuristic_mask, "heuristic", "no weights yet (fallback)"