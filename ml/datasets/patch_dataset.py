# ml/datasets/patch_dataset.py
"""
Full-resolution patch dataset for bryozoan segmentation.

Each full-resolution image is tiled into 512×512 patches with 25% overlap.
Patches are filtered by blade-mask intersection so background-only tiles
are skipped. Images without a blade mask contribute all patches.

CV splits must be on SOURCE IMAGES, not patches, to prevent data leakage.
This class takes a pre-filtered list of image paths (train or val side of a
split) and generates all qualifying patches from those images only.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.datasets.bryozoan_dataset import _build_transform, _IMAGENET_MEAN_T, _IMAGENET_STD_T


def _patch_starts(total: int, patch: int, step: int) -> list[int]:
    """Generate patch start positions, clamping the last patch to stay in bounds."""
    starts = list(range(0, total - patch + 1, step))
    if not starts or starts[-1] + patch < total:
        starts.append(total - patch)
    return starts


class PatchDataset(Dataset):
    """
    Args:
        image_paths:   List of Path objects for the images in this split.
        masks_dir:     Directory containing GT masks (*_bry_gt.png).
        blade_dir:     Directory containing blade masks (*_blade_mask.png).
                       Images without a blade mask have all patches included.
        patch_size:    Spatial size of each square patch (default 512).
        overlap:       Fractional overlap between adjacent patches (default 0.25).
        min_blade_px:  Minimum blade pixels in a patch to keep it (default 1000).
        augment:       If True, apply albumentations augmentation.
        imagenet_norm: If True, normalise to ImageNet mean/std after /255.
        use_clahe:     Include CLAHE in the augmentation pipeline.
        rng_seed:      Seed for augmentation RNG.
    """

    def __init__(
        self,
        image_paths: list,
        masks_dir: Path,
        blade_dir: Path,
        patch_size: int = 512,
        overlap: float = 0.25,
        min_blade_px: int = 1000,
        augment: bool = False,
        imagenet_norm: bool = False,
        use_clahe: bool = False,
        rng_seed: int | None = None,
    ):
        self.patch_size    = patch_size
        self.augment       = augment
        self.imagenet_norm = imagenet_norm
        self._rng          = np.random.default_rng(rng_seed)
        self._transform    = _build_transform(use_clahe) if augment else None

        step = int(patch_size * (1.0 - overlap))

        # Pre-load images and masks into memory for fast __getitem__.
        # With 24 images at ~29 MB each, peak RAM usage is ~700 MB per fold.
        self._imgs   = {}   # stem -> (H, W, 3) uint8 RGB
        self._masks  = {}   # stem -> (H, W) uint8 {0, 1}
        self._patches = []  # list of (stem, y0, x0)

        for p in image_paths:
            stem = p.stem
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            self._imgs[stem] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp = masks_dir / f"{stem}_bry_gt.png"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                self._masks[stem] = (m > 0).astype(np.uint8) if m is not None else None
            else:
                self._masks[stem] = None

            h, w = self._imgs[stem].shape[:2]

            # Load blade mask for patch filtering
            bp = blade_dir / f"{stem}_blade_mask.png"
            blade = cv2.imread(str(bp), cv2.IMREAD_GRAYSCALE) if bp.exists() else None

            for y0 in _patch_starts(h, patch_size, step):
                for x0 in _patch_starts(w, patch_size, step):
                    if blade is not None:
                        roi = blade[y0:y0 + patch_size, x0:x0 + patch_size]
                        if int((roi > 0).sum()) < min_blade_px:
                            continue
                    self._patches.append((stem, y0, x0))

    def __len__(self):
        return len(self._patches)

    def _augment_patch(self, img: np.ndarray, mask: np.ndarray):
        seed      = int(self._rng.integers(2**31))
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            result = self._transform(image=img, mask=mask)
        finally:
            np.random.set_state(rng_state)
        return result["image"], result["mask"]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        if self.imagenet_norm:
            t = (t - _IMAGENET_MEAN_T) / _IMAGENET_STD_T
        return t

    def __getitem__(self, idx: int):
        stem, y0, x0 = self._patches[idx]
        ps  = self.patch_size
        img  = self._imgs[stem][y0:y0 + ps, x0:x0 + ps].copy()
        raw_mask = self._masks.get(stem)
        mask = raw_mask[y0:y0 + ps, x0:x0 + ps].copy() if raw_mask is not None \
               else np.zeros((ps, ps), dtype=np.uint8)

        if self.augment:
            img, mask = self._augment_patch(img, mask)

        return {
            "image":    self._to_tensor(img),
            "mask":     torch.from_numpy(mask).float().unsqueeze(0),
            "stem":     f"{stem}_{y0}_{x0}",
            "has_mask": raw_mask is not None,
        }
