from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.models.model_factory import IMAGENET_MEAN, IMAGENET_STD


def _read_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return (m > 0).astype(np.uint8)  # 0/1


_IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD_T  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)


def _build_transform(use_clahe: bool) -> A.Compose:
    """
    Albumentations pipeline applied during training.

    Spatial transforms (applied identically to image + mask):
      HorizontalFlip, VerticalFlip, RandomRotate90 (uniform over 0/90/180/270°)

    Photometric transforms (image only):
      RandomBrightnessContrast — replicates old alpha/beta jitter
      HueSaturationValue       — colour-space augmentation for kelp/bryozoan distinction
      GaussNoise               — mild sensor noise (std≈3–7 px, equivalent to var 10–50)
      CLAHE                    — optional local contrast normalisation (config flag)
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),          # uniform over {0°, 90°, 180°, 270°}
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        A.GaussNoise(std_range=(0.012, 0.028), p=0.2),  # equiv. var_limit=(10,50) on 0-255
    ]
    if use_clahe:
        transforms.append(A.CLAHE(clip_limit=2.0, p=0.3))
    return A.Compose(transforms)


class BryozoanDataset(Dataset):
    """
    Dataset for bryozoan segmentation.

    Images:
      data/raw_for_annot/<batch>/*.(jpg/png/tif)

    GT Masks (optional):
      data/processed_annot/<batch>/bryozoan_gt_masks/<stem>_bry_gt.png

    Args:
        images_dir:      Directory containing raw images.
        masks_dir:       Directory containing GT masks. None = unlabeled mode.
        size:            Resize target (square, divisible by 32 for smp models).
        augment:         If True, apply albumentations augmentation pipeline.
        indices:         Optional subset indices for train/val splitting.
        rng_seed:        Seed for the augmentation RNG (reproducibility).
        imagenet_norm:   If True, normalise to ImageNet mean/std after /255.
                         Must be True when using pretrained smp encoders.
        use_clahe:       If True, add CLAHE to the augmentation pipeline.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path | None = None,
        size: int = 512,
        augment: bool = False,
        indices: list | None = None,
        rng_seed: int | None = None,
        imagenet_norm: bool = False,
        use_clahe: bool = False,
    ):
        self.size          = size
        self.masks_dir     = masks_dir
        self.augment       = augment
        self.imagenet_norm = imagenet_norm
        self._rng          = np.random.default_rng(rng_seed)
        self._transform    = _build_transform(use_clahe) if augment else None

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        all_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

        if masks_dir is None:
            candidates = all_images
        else:
            candidates = [
                p for p in all_images
                if (masks_dir / f"{p.stem}_bry_gt.png").exists()
                and (masks_dir / f"{p.stem}_bry_gt.png").stat().st_size > 0
            ]

        self.images = [candidates[i] for i in indices] if indices is not None else candidates

    def __len__(self):
        return len(self.images)

    def _augment(self, img: np.ndarray, mask: np.ndarray | None):
        """
        Run the albumentations pipeline.

        Reproducibility: derive a per-call seed from the dataset's seeded RNG,
        apply it to numpy's global state (which albumentations uses internally),
        then restore the previous state so other code is unaffected.
        """
        seed      = int(self._rng.integers(2**31))
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            dummy = np.zeros(img.shape[:2], dtype=np.uint8)
            result = self._transform(image=img, mask=mask if mask is not None else dummy)
        finally:
            np.random.set_state(rng_state)

        out_img  = result["image"]
        out_mask = result["mask"] if mask is not None else None
        return out_img, out_mask

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        if self.imagenet_norm:
            t = (t - _IMAGENET_MEAN_T) / _IMAGENET_STD_T
        return t

    def __getitem__(self, idx: int):
        p    = self.images[idx]
        stem = p.stem

        img = _read_img(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)

        if self.masks_dir is None:
            if self.augment:
                img, _ = self._augment(img, None)
            return {"image": self._to_tensor(img), "stem": stem, "has_mask": False}

        mp = self.masks_dir / f"{stem}_bry_gt.png"
        if not mp.exists():
            if self.augment:
                img, _ = self._augment(img, None)
            return {"image": self._to_tensor(img), "stem": stem, "has_mask": False}

        mask = _read_mask(mp)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        return {
            "image":    self._to_tensor(img),
            "mask":     torch.from_numpy(mask).float().unsqueeze(0),
            "stem":     stem,
            "has_mask": True,
        }
