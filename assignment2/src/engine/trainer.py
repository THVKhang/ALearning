
import csv
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.loss import MaskedSILogLoss
from src.engine.metrics import DepthMetricTracker


class Trainer:
    """
    Depth Estimation Training Pipeline:
    - Supports PyTorch AMP (Automatic Mixed Precision with fp16)
    - Evaluates using standard depth estimation metrics (Abs Rel, RMSE, Delta)
    - Saves the best model checkpoint based on validation Abs Rel
    - Logs training and validation history to a CSV file for plotting
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        amp: bool = True,
        output_dir: str = "outputs/baseline_unet",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.output_dir = output_dir

        # Enable AMP GradScaler only when using a CUDA GPU
        self.amp = amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "history.csv")
        self._init_csv()

        self.best_val_abs_rel = float("inf")

    def _init_csv(self):
        """Initialize the CSV file for recording training and validation history."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_abs_rel",
                "val_rmse",
                "val_delta_1",
                "val_delta_2",
                "val_delta_3",
                "lr",
                "epoch_time_sec",
            ])

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            depth = batch["depth"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Run the forward pass using mixed precision (fp16)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp,
            ):
                pred = self.model(rgb)
                loss = self.criterion(pred, depth, mask)

            # Perform a backward pass safely using GradScaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / max(n_batches, 1)

    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)
        tracker = DepthMetricTracker()

        with torch.no_grad():
            for batch in self.val_loader:
                rgb = batch["rgb"].to(self.device, non_blocking=True)
                depth = batch["depth"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                domains = batch.get("domain", None)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    pred = self.model(rgb)
                    loss = self.criterion(pred, depth, mask)

                total_loss += loss.item()
                tracker.update(pred, depth, mask, domains)

        summary = tracker.summary()
        summary["overall"]["val_loss"] = total_loss / max(n_batches, 1)
        return summary

    def train(self, epochs: int):
        print(
            f"\n[Trainer] Starting model training on device: "
            f"{self.device} (AMP={self.amp})"
        )
        print(
            f"[Trainer] Checkpoints and logs will be saved to: "
            f"{self.output_dir}\n"
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # 1. Training
            train_loss = self.train_one_epoch(epoch)

            # 2. Validation
            val_summary = self.validate()
            ov = val_summary["overall"]
            val_loss = ov["val_loss"]
            val_abs_rel = ov["abs_rel"]
            val_rmse = ov["rmse"]
            val_d1 = ov["delta_1"]

            # Update the learning rate scheduler, if available
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            # 3. Print metrics for the current epoch
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"AbsRel: {val_abs_rel:.4f} | "
                f"RMSE: {val_rmse:.2f}m | "
                f"d1: {val_d1:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # 4. Append the training history to the CSV file
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_loss,
                    val_loss,
                    val_abs_rel,
                    val_rmse,
                    val_d1,
                    ov["delta_2"],
                    ov["delta_3"],
                    current_lr,
                    round(epoch_time, 2),
                ])

            # 5. Save the best checkpoint when validation Abs Rel improves
            # Lower Abs Rel indicates better performance
            if val_abs_rel < self.best_val_abs_rel:
                self.best_val_abs_rel = val_abs_rel
                best_path = os.path.join(
                    self.output_dir, "best_model.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_abs_rel": val_abs_rel,
                        "val_summary": val_summary,
                    },
                    best_path,
                )

                print(
                    f"  -> Saved a new best checkpoint "
                    f"(Abs Rel: {val_abs_rel:.4f}) to {best_path}"
                )

        print(
            f"\n[Trainer] Training completed after {epochs} epochs! "
            f"Best Abs Rel: {self.best_val_abs_rel:.4f}"
        )
