import csv
import os
import time

import torch

from src.engine.align import align_batch
from src.engine.metrics import MetricTracker


class Trainer:
    def __init__(self, model, output_type, criterion, optimizer, scheduler, loaders, cfg,
                 device="cuda", max_batches=None):
        self.model = model.to(device)
        self.output_type = output_type
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.device = device
        self.amp = cfg["train"]["amp"] and device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.out_dir = cfg["out_dir"]
        self.budget = cfg["train"].get("time_budget_min")
        self.max_batches = max_batches  # smoke test only
        os.makedirs(self.out_dir, exist_ok=True)

    def _to_device(self, batch):
        return (batch["rgb"].to(self.device, non_blocking=True),
                batch["depth"].to(self.device, non_blocking=True),
                batch["mask"].to(self.device, non_blocking=True))

    def train_one_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for i, batch in enumerate(self.train_loader):
            if self.max_batches and i >= self.max_batches:
                break
            rgb, depth, mask = self._to_device(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                pred = self.model(rgb)
            loss = self.criterion(pred.float(), depth, mask)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        tracker = MetricTracker()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if self.max_batches and i >= self.max_batches:
                break
            rgb, depth, mask = self._to_device(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                raw = self.model(rgb)
            raw = raw.float()
            total += self.criterion(raw, depth, mask).item()
            n += 1
            pred = raw if self.output_type == "metric" else align_batch(raw, depth, mask, batch["domain"])
            tracker.update(pred, depth, mask, batch["domain"])
        res = tracker.summary()
        res["loss"] = total / max(n, 1)
        return res

    def fit(self, epochs):
        cols = ["epoch", "train_loss", "val_loss"]
        for dom in ("overall", "indoor", "outdoor"):
            cols += [f"{dom}_abs_rel", f"{dom}_rmse", f"{dom}_delta1"]
        cols += ["lr", "epoch_time_s", "peak_vram_gb"]
        hist_path = os.path.join(self.out_dir, "history.csv")
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

        best, best_epoch, start = float("inf"), 0, time.time()
        if self.device == "cuda":
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
            vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if self.device == "cuda" else 0.0

            row = [epoch, train_loss, val["loss"]]
            for dom in ("overall", "indoor", "outdoor"):
                row += [val[dom]["abs_rel"], val[dom]["rmse"], val[dom]["delta1"]]
            row += [lr, round(dt, 1), round(vram, 3)]
            with open(hist_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            ov = val["overall"]
            print(f"epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val['loss']:.4f} | "
                  f"AbsRel {ov['abs_rel']:.4f} RMSE {ov['rmse']:.3f} d1 {ov['delta1']:.3f} | "
                  f"{dt:.0f}s {vram:.2f}GB")

            # checkpoint rule: lowest overall val Abs Rel
            if ov["abs_rel"] < best:
                best, best_epoch = ov["abs_rel"], epoch
                torch.save({"epoch": epoch, "model": self.model.state_dict(), "val": val},
                           os.path.join(self.out_dir, "best.pth"))
                print(f"  -> new best (AbsRel {best:.4f}), saved best.pth")

            elapsed = (time.time() - start) / 60
            if self.budget and elapsed + max(epoch_times) / 60 > self.budget:
                print(f"time budget {self.budget} min reached after {epoch} epochs")
                break

        return {
            "best_epoch": best_epoch,
            "best_val_abs_rel": best,
            "epochs_run": len(epoch_times),
            "train_time_min": (time.time() - start) / 60,
            "epoch_time_s": sum(epoch_times) / len(epoch_times),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if self.device == "cuda" else 0.0,
        }
