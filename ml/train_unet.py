from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ml.models.unet import UNetSmall
from ml.datasets.bryozoan_dataset import BryozoanDataset


def main():
    ROOT = Path(__file__).resolve().parents[1]
    batch = "bryozoan_batch_01"

    images_dir = ROOT / "data" / "raw_for_annot" / batch
    masks_dir = ROOT / "data" / "processed_annot" / batch / "bryozoan_gt_masks"

    # If folder doesn't exist or is empty, exit cleanly
    if not masks_dir.exists():
        print(f"[INFO] GT masks folder does not exist: {masks_dir}")
        print("Create it and add masks named <stem>_bry_gt.png, then rerun.")
        return

    ds = BryozoanDataset(images_dir, masks_dir=masks_dir, size=512)

    if len(ds) == 0:
        print(f"[INFO] No usable GT masks found in: {masks_dir}")
        print("Add PNG masks named <stem>_bry_gt.png (matching image stems), then rerun training.")
        return

    print(f"[INFO] Training on {len(ds)} labeled images (image+mask pairs).")

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNetSmall().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simple baseline training loop
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0

        for batch in dl:
            # Some safety: only train batches that actually contain masks
            if "mask" not in batch:
                continue

            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        avg = loss_sum / max(len(dl), 1)
        print(f"Epoch {epoch}/{epochs}: loss={avg:.4f}")

    ckpt = ROOT / "models" / "bry_unet.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
