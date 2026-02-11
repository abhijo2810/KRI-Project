from pathlib import Path
import cv2
import numpy as np
import torch
from ml.models.unet import UNetSmall

def infer_one(img_bgr: np.ndarray, model, device="cpu", size=512, thr=0.5):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    m = (prob > thr).astype(np.uint8) * 255
    return cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

def main():
    ROOT = Path(__file__).resolve().parents[1]
    ckpt = ROOT / "models" / "bry_unet.pt"
    if not ckpt.exists():
        print("[INFO] No trained model yet. This script is ready once labels exist.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("Loaded model:", ckpt)

if __name__ == "__main__":
    main()
