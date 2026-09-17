"""Figures for the report: training curves, confusion matrix, class counts, samples.

Colors are the Okabe-Ito colorblind-safe pair for the two-series curves, and a
single-hue sequential ramp for the confusion matrix. Train/val are also separated by
line style so the series stay distinguishable in grayscale print.
"""

import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "#0072B2"
VAL_COLOR = "#D55E00"
BAR_COLOR = "#0072B2"
SEQUENTIAL_CMAP = "Blues"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_curves(history, out_path, title=None):
    """Loss and macro-F1 against epoch, train vs validation, one panel per measure."""
    epochs = [row["epoch"] for row in history]
    panels = [
        ("loss", "train_loss", "val_loss", "Cross-entropy loss"),
        ("macro_f1", "train_macro_f1", "val_macro_f1", "Macro-F1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (_, train_key, val_key, label) in zip(axes, panels):
        ax.plot(epochs, [row[train_key] for row in history], color=TRAIN_COLOR,
                linewidth=2, marker="o", markersize=4, label="train")
        ax.plot(epochs, [row[val_key] for row in history], color=VAL_COLOR,
                linewidth=2, linestyle="--", marker="s", markersize=4, label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(label, fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        _style(ax)

    if title:
        fig.suptitle(title, fontsize=12, color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm, class_names, out_path, normalize=True, title=None):
    """Confusion matrix heatmap; normalized rows read as per-class recall."""
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = ax.imshow(shown, cmap=SEQUENTIAL_CMAP, vmin=0,
                      vmax=1 if normalize else shown.max())

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title, fontsize=11)

    threshold = shown.max() * 0.6
    for row in range(shown.shape[0]):
        for col in range(shown.shape[1]):
            value = shown[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > threshold else INK)

    bar = fig.colorbar(image, ax=ax, fraction=0.046)
    bar.set_label("recall" if normalize else "count", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_class_distribution(counts, class_names, out_path, title=None):
    """Samples per class for one split, with the count written on each bar."""
    counts = list(counts)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = ax.bar(range(len(counts)), counts, color=BAR_COLOR, width=0.72)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_ylabel("samples")
    ax.set_ylim(0, max(counts) * 1.15)
    if title:
        ax.set_title(title, fontsize=11)
    for bar_patch, count in zip(bars, counts):
        ax.text(bar_patch.get_x() + bar_patch.get_width() / 2, count, f"{count:,}",
                ha="center", va="bottom", fontsize=8, color=INK)
    _style(ax)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sample_grid(images, true_labels, class_names, out_path, pred_labels=None,
                     ncols=8, title=None):
    """Grayscale image grid; captions show the true label, or `true -> pred` on errors."""
    images = np.asarray(images)
    count = len(images)
    nrows = int(np.ceil(count / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols, 1.6 * nrows),
                             constrained_layout=True)
    for position, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.axis("off")
        if position >= count:
            continue
        ax.imshow(np.squeeze(images[position]), cmap="gray", vmin=0, vmax=1)
        caption = class_names[int(true_labels[position])]
        if pred_labels is not None:
            caption = f"{caption}\n-> {class_names[int(pred_labels[position])]}"
        ax.set_title(caption, fontsize=7.5, color=INK)

    if title:
        fig.suptitle(title, fontsize=12, color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
