
"""Training script for Baseline U-Net and Pretrained Models on DIODE."""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Import modules from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.dataset import get_dataloaders, DiodeDepthDataset
from src.models.unet import UNet
from src.engine.loss import MaskedSILogLoss
from src.engine.trainer import Trainer


def set_seed(seed: int = 42):
    """Set random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Train a Monocular Depth Estimation Model"
    )
    parser.add_argument(
        "--config",
        default="configs/baseline_unet.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override the dataset root directory if needed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the batch size",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick test (1 epoch on a small subset)",
    )
    args = parser.parse_args()

    # 1. Load the YAML configuration file
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override configuration parameters with command-line arguments
    data_root = args.data_root or cfg["data"]["root"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    seed = cfg["data"].get("seed", 42)

    set_seed(seed)

    print("=" * 60)
    print(f"  STARTING TRAINING: {cfg['experiment_name']}")
    print(f"  Dataset: {data_root} | Seed: {seed} | Batch Size: {batch_size}")
    print("=" * 60)

    # 2. Initialize DataLoaders
    # In smoke-test mode, use only 16 training samples and 8 validation samples
    # to verify that the training pipeline works correctly.
    if args.smoke:
        print("[Smoke Test] Quick test mode enabled...")
        epochs = 1
        output_dir = "outputs/smoke_test"

        train_ds = DiodeDepthDataset(data_root, split="train")
        val_ds = DiodeDepthDataset(data_root, split="val")

        train_subset = Subset(
            train_ds, list(range(min(16, len(train_ds))))
        )
        val_subset = Subset(
            val_ds, list(range(min(8, len(val_ds))))
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        output_dir = cfg.get("output_dir", "outputs/baseline_unet")
        loaders = get_dataloaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg["data"].get("num_workers", 2),
            augment=cfg["data"].get("augment", True),
            seed=seed,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

    # 3. Initialize the U-Net model
    model = UNet(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 1),
    )

    # 4. Define the loss function and optimizer
    criterion = MaskedSILogLoss(
        lambda_factor=cfg["training"].get("lambda_factor", 0.5)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    # 5. Initialize the trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=cfg["training"].get("amp", True),
        output_dir=output_dir,
    )

    trainer.train(epochs=epochs)


if __name__ == "__main__":
    main()
