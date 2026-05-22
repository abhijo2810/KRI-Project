# ml/infer_unet.py
"""
Batch inference script: runs the trained UNet over all images in the raw
directory and writes predicted bryozoan masks alongside QA overlays.

Run from the project root:
    python -m ml.infer_unet
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from ml.models.unet import UNetSmall

INFER_SIZE = 512


def _load_model(ckpt: Path, device: str) -> UNetSmall:
    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def infer_one(img_bgr: np.ndarray, model, device: str, thr: float = 0.5) -> np.ndarray:
    """Return a binary mask (0/255, original resolution) for one image."""
    h, w = img_bgr.shape[:2]
    img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x    = cv2.resize(img, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
    x    = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x    = x.to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    m = (prob > thr).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    ROOT        = Path(__file__).resolve().parents[1]
    BATCH_NAME  = "bryozoan_batch_01"
    ckpt        = ROOT / "models" / "bry_unet.pt"
    raw_dir     = ROOT / "data" / "raw_for_annot" / BATCH_NAME
    out_dir     = ROOT / "outputs" / "overlays_ml_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        print(f"[INFO] No trained model at {ckpt}. Run python -m ml.train_unet first.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = _load_model(ckpt, device)
    print(f"Loaded model: {ckpt}  device={device}")

    exts   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths  = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in exts])

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] unreadable: {p.name}")
            continue
        mask = infer_one(img, model, device)
        overlay = img.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 0, 128]) * 0.5).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{p.stem}_ml.png"), overlay)
        print(f"[OK] {p.stem}")

    print(f"\nOverlays written to: {out_dir}")


if __name__ == "__main__":
    main()
