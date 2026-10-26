
"""End-to-end pipeline verification using a synthetic mini dataset."""

import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.unet import UNet
from src.engine.loss import MaskedSILogLoss
from src.engine.trainer import Trainer


class SyntheticDiodeDataset(Dataset):
    """A synthetic dataset containing 8 samples in the DIODE-compatible format."""

    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # RGB image: normalized tensor with shape (3, 288, 384)
        rgb = torch.randn(3, 288, 384)

        # Depth map: random distances ranging from 1 m to 10 m
        depth = torch.rand(1, 288, 384) * 9.0 + 1.0

        # Mask: 85% valid pixels and 15% invalid pixels (0)
        mask = (torch.rand(1, 288, 384) > 0.15).float()

        domain = "indoor" if idx % 2 == 0 else "outdoor"

        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "domain": domain,
        }


def test_pipeline():
    print("=" * 60)
    print("  END-TO-END TRAINING PIPELINE VERIFICATION (TASK 2)")
    print("=" * 60)

    output_dir = "outputs/test_synthetic_run"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    train_ds = SyntheticDiodeDataset(n_samples=8)
    val_ds = SyntheticDiodeDataset(n_samples=4)

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
    )

    model = UNet(in_channels=3, out_channels=1)
    criterion = MaskedSILogLoss(lambda_factor=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        amp=torch.cuda.is_available(),
        output_dir=output_dir,
    )

    # Run a test with 2 training epochs
    print("[1/3] Running 2 training and validation epochs...")
    trainer.train(epochs=2)

    # Verify the history.csv file
    print("[2/3] Checking the history.csv file...")
    csv_file = os.path.join(output_dir, "history.csv")

    assert os.path.exists(csv_file), "Error: history.csv was not found!"

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(
            f"  -> The history.csv file successfully recorded "
            f"{len(lines) - 1} epochs."
        )
        assert len(lines) == 3, (
            "Error: The CSV file does not contain records for both epochs!"
        )

    # Verify the best_model.pth checkpoint file
    print("[3/3] Checking the best_model.pth checkpoint file...")
    ckpt_file = os.path.join(output_dir, "best_model.pth")

    assert os.path.exists(ckpt_file), (
        "Error: best_model.pth was not found!"
    )

    ckpt = torch.load(ckpt_file, weights_only=False)

    assert "model_state_dict" in ckpt, (
        "Error: The checkpoint does not contain the model weights!"
    )

    print(
        f"  -> Checkpoint loaded successfully: "
        f"best_abs_rel = {ckpt['val_abs_rel']:.4f}"
    )

    print("\n" + "=" * 60)
    print(
        ">>> CONGRATULATIONS: THE TRAINING PIPELINE "
        "HAS COMPLETED THE TEST SUCCESSFULLY! <<<"
    )
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
