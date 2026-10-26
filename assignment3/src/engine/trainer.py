import csv
import os
import time

import torch

from src.utils.metrics import compute_classification_metrics


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, loaders, cfg, device="cuda", max_batches=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.amp = cfg["train"].get("amp", True) and self.is_cuda
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.out_dir = cfg["out_dir"]
        self.max_batches = max_batches
        os.makedirs(self.out_dir, exist_ok=True)

    def _to_device(self, batch):
        return {
            "rgb": batch["rgb"].to(self.device, non_blocking=True),
            "depth_geo": batch["depth_geo"].to(self.device, non_blocking=True),
            "valid_ratio": batch["valid_ratio"].to(self.device, non_blocking=True),
            "label": batch["label"].to(self.device, non_blocking=True),
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        for i, batch in enumerate(self.train_loader):
            if self.max_batches and i >= self.max_batches:
                break
            b = self._to_device(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                out = self.model(b)
                loss = self.criterion(out, b["label"])

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        y_true, y_pred = [], []
        for i, batch in enumerate(self.val_loader):
            if self.max_batches and i >= self.max_batches:
                break
            b = self._to_device(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                out = self.model(b)
                loss = self.criterion(out, b["label"])
            total_loss += loss.item()
            n += 1
            preds = out["logits"].argmax(dim=-1)
            y_true.extend(b["label"].cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(n, 1)
        return metrics

    def fit(self, epochs):
        cols = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_domain_acc",
            "val_elev_acc",
            "lr",
            "epoch_time_s",
            "peak_vram_gb",
        ]
        hist_path = os.path.join(self.out_dir, "history.csv")
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

        best_f1, best_epoch = -1.0, 0
        start = time.time()
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        epoch_times = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch()
            val = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()
            dt = time.time() - t0
            epoch_times.append(dt)
            vram = torch.cuda.max_memory_allocated() / 1024**3 if self.is_cuda else 0.0

            row = [
                epoch,
                round(train_loss, 5),
                round(val["loss"], 5),
                round(val["accuracy"], 5),
                round(val["macro_f1"], 5),
                round(val["domain_accuracy"], 5),
                round(val["elevation_accuracy"], 5),
                lr,
                round(dt, 2),
                round(vram, 3),
            ]
            with open(hist_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            print(
                f"epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val['loss']:.4f} | "
                f"Acc {val['accuracy']:.4f} MacroF1 {val['macro_f1']:.4f} "
                f"(Dom {val['domain_accuracy']:.3f}, Elev {val['elevation_accuracy']:.3f}) | {dt:.1f}s"
            )

            # Checkpoint selection rule: highest validation macro-F1
            if val["macro_f1"] > best_f1:
                best_f1, best_epoch = val["macro_f1"], epoch
                torch.save(
                    {"epoch": epoch, "model": self.model.state_dict(), "val": val},
                    os.path.join(self.out_dir, "best.pth"),
                )

        return {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "epochs_run": len(epoch_times),
            "train_time_s": time.time() - start,
            "epoch_time_s": sum(epoch_times) / max(len(epoch_times), 1),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if self.is_cuda else 0.0,
        }
