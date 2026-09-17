"""Train/validation/test loops shared by every model in the comparison."""

import time

import numpy as np
import torch
from torch import nn

from src.utils.metrics import classification_metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`: trains when `optimizer` is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Weight by batch size: the final batch is usually smaller than the rest.
            total_loss += loss.item() * targets.size(0)
            y_true.append(targets.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return metrics, y_true, y_pred


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def fit(model, train_loader, val_loader, device, epochs, optimizer="adam", lr=1e-3,
        weight_decay=0.0, monitor="macro_f1", early_stopping_patience=None,
        checkpoint_path=None, log_fn=print):
    """Train for `epochs`, keeping the checkpoint with the best validation `monitor`."""
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, optimizer, lr, weight_decay)

    history = []
    best_score, best_epoch, stale_epochs = -float("inf"), 0, 0
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics, *_ = run_epoch(model, train_loader, criterion, device, opt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        total_train_time += epoch_time

        val_metrics, *_ = run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": epoch_time,
        })

        improved = val_metrics[monitor] > best_score
        if improved:
            best_score, best_epoch, stale_epochs = val_metrics[monitor], epoch, 0
            if checkpoint_path is not None:
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "monitor": monitor,
                    "val_metrics": val_metrics,
                }, checkpoint_path)
        else:
            stale_epochs += 1

        log_fn(
            f"epoch {epoch:3d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"macroF1 {val_metrics['macro_f1']:.4f} | {epoch_time:.1f}s"
            + ("  <- best" if improved else "")
        )

        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            log_fn(f"early stopping: no val {monitor} gain for {stale_epochs} epochs")
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "monitor": monitor,
        "epochs_run": len(history),
        "train_seconds_total": total_train_time,
        "train_seconds_per_epoch": total_train_time / max(len(history), 1),
    }


@torch.no_grad()
def measure_inference_time(model, loader, device, repeats=3):
    """Median wall-clock time for one full forward pass over `loader`."""
    model.eval()
    for inputs, _ in loader:  # warm up kernels / allocator before timing
        model(inputs.to(device))
        break

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for inputs, _ in loader:
            model(inputs.to(device, non_blocking=True))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    total = float(np.median(times))
    num_images = len(loader.dataset)
    return {
        "total_seconds": total,
        "ms_per_image": 1000.0 * total / num_images,
        "num_images": num_images,
        "batch_size": loader.batch_size,
    }
