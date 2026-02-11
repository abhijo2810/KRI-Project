from pathlib import Path
from ml.datasets.bryozoan_dataset import BryozoanDataset

ROOT = Path(__file__).resolve().parents[1]
batch = "bryozoan_batch_01"
images_dir = ROOT / "data" / "raw_for_annot" / batch

ds = BryozoanDataset(images_dir, masks_dir=None, size=512)
item = ds[0]
print("len:", len(ds))
print("keys:", item.keys())
print("stem:", item["stem"], "has_mask:", item["has_mask"])
print("image shape:", item["image"].shape)
