from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read mask: {path}")
    m = (m > 0).astype(np.uint8)  # 0/1
    return m


class BryozoanDataset(Dataset):
    """
    Dataset for bryozoan segmentation.

    Images:
      data/raw_for_annot/<batch>/*.(jpg/png/tif)

    GT Masks (optional):
      data/processed_annot/<batch>/bryozoan_gt_masks/<stem>_bry_gt.png

    IMPORTANT:
    - If masks_dir is provided, we ONLY include images that have a matching mask.
    - This prevents crashes when you have partial labels (very common during annotation).
    """

    def __init__(self, images_dir: Path, masks_dir: Path | None = None, size: int = 512):
        self.size = size
        self.masks_dir = masks_dir

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        all_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

        if masks_dir is None:
            self.images = all_images
        else:
            filtered = []
            for p in all_images:
                mp = masks_dir / f"{p.stem}_bry_gt.png"
                if mp.exists() and mp.stat().st_size > 0:
                    filtered.append(p)
            self.images = filtered

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        p = self.images[idx]
        stem = p.stem

        img = _read_img(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3,H,W)

        # Unlabeled mode
        if self.masks_dir is None:
            return {"image": img_t, "stem": stem, "has_mask": False}

        # Labeled mode (safe)
        mp = self.masks_dir / f"{stem}_bry_gt.png"
        if not mp.exists():
            # Shouldn't happen because we filtered, but keep it safe
            return {"image": img_t, "stem": stem, "has_mask": False}

        mask = _read_mask(mp)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # (1,H,W)

        return {"image": img_t, "mask": mask_t, "stem": stem, "has_mask": True}
