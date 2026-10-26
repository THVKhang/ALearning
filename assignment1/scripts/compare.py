"""Gather test metrics and training histories across all Assignment-1 runs.

Prints the four comparison tables reported in README.md and index.html (overall metrics,
per-class F1, upper-body vs. other-six F1 gap, and error-group breakdown) directly from the
committed JSON artifacts under outputs/, and optionally regenerates each run's training
curves and test confusion-matrix figures.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plots import plot_confusion_matrix, plot_curves  # noqa: E402

MODELS = [
    ("linear", "Linear / softmax"),
    ("mlp", "MLP (784-256-10, dropout 0.2)"),
    ("cnn", "CNN (32-64 conv-pool, dropout 0.2)"),
    ("lstm", "LSTM (28 row steps, hidden 128)"),
    ("gru", "GRU (28 row steps, hidden 128)"),
    ("transformer", "Transformer (28 row tokens, d_model 128, 4 heads, 2 layers)"),
]

UPPER_BODY = ["Shirt", "Pullover", "Coat", "T-shirt/top"]
OTHER_SIX = ["Dress", "Sneaker", "Sandal", "Bag", "Ankle boot", "Trouser"]
CLASS_ORDER_BY_LINEAR_DIFFICULTY = UPPER_BODY + OTHER_SIX


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true",
                        help="regenerate curves.png and confusion_matrix_test.png in each run dir")
    parser.add_argument("--export-assets", action="store_true",
                        help="also copy generated curves and confusion matrices into assets/")
    return parser.parse_args()


def load_runs(outputs_dir, dataset, seed):
    runs = {}
    for key, label in MODELS:
        run_dir = Path(outputs_dir) / f"{key}_{dataset}_seed{seed}"
        if not run_dir.exists():
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        runs[key] = {
            "label": label,
            "dir": run_dir,
            "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
            "history": json.loads((run_dir / "history.json").read_text(encoding="utf-8")),
            "metrics": json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8")),
        }
    return runs


def format_main_table(runs):
    lines = [
        "| Model | Params | Test accuracy | Test macro-F1 | Best epoch | Train time | Inference, model only | Inference, end to end |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, _ in MODELS:
        run = runs[key]
        metrics = run["metrics"]
        history = run["history"]
        inf = metrics["inference"]
        epochs_run = history["epochs_run"]
        best_ep = metrics["checkpoint_epoch"]
        total_s = history["train_seconds_total"]
        per_ep_s = history["train_seconds_per_epoch"]
        lines.append(
            f"| {run['label']} | {metrics['parameters']['total']:,} | "
            f"{metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{best_ep} / {epochs_run} | {total_s:.1f} s ({per_ep_s:.1f} s/epoch) | "
            f"{inf['ms_per_image']:.4f} ms/image | {inf['end_to_end_ms_per_image']:.4f} ms/image |"
        )
    return "\n".join(lines)


def format_per_class_table(runs):
    short_names = ["Linear", "MLP", "CNN", "LSTM", "GRU", "Transformer"]
    lines = [
        "| Class | " + " | ".join(short_names) + " |",
        "|---|" + "|".join(["---"] * len(short_names)) + "|",
    ]
    for cls_name in CLASS_ORDER_BY_LINEAR_DIFFICULTY:
        scores = [runs[key]["metrics"]["per_class_f1"][cls_name] for key, _ in MODELS]
        best = max(scores)
        cells = [
            f"**{s:.4f}**" if abs(s - best) < 1e-9 else f"{s:.4f}"
            for s in scores
        ]
        lines.append(f"| {cls_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_cluster_gap_table(runs):
    lines = [
        "| Model | Mean F1, upper-body 4 | Mean F1, other 6 | Gap |",
        "|---|---|---|---|",
    ]
    for key, _ in MODELS:
        f1_map = runs[key]["metrics"]["per_class_f1"]
        upper = float(np.mean([f1_map[c] for c in UPPER_BODY]))
        other = float(np.mean([f1_map[c] for c in OTHER_SIX]))
        gap = other - upper
        lines.append(f"| {key.upper() if key in ('mlp', 'cnn', 'lstm', 'gru') else runs[key]['label'].split(' (')[0]} "
                     f"| {upper:.4f} | {other:.4f} | {gap:.4f} |")
    return "\n".join(lines)


def format_error_breakdown_table(runs):
    short_names = ["Linear", "MLP", "CNN", "LSTM", "GRU", "Transformer"]
    lines = [
        "| Error group | " + " | ".join(short_names) + " |",
        "|---|" + "|".join(["---"] * len(short_names)) + "|",
    ]

    classes = runs["linear"]["config"]["data"]["classes"]
    idx = {name: i for i, name in enumerate(classes)}
    ub_indices = [idx[c] for c in ["T-shirt/top", "Pullover", "Coat", "Shirt"]]
    clothing_indices = [idx[c] for c in ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Shirt"]]
    non_clothing_indices = [idx[c] for c in ["Sandal", "Sneaker", "Bag", "Ankle boot"]]

    totals = {}
    stats = {
        "Shirt -> T-shirt/top": {},
        "T-shirt/top -> Shirt": {},
        "Pullover <-> Coat": {},
        "Shirt -> Pullover / Coat": {},
        "All confusions inside {T-shirt/top, Pullover, Coat, Shirt}": {},
        "Clothing <-> footwear / bag (cross-group)": {},
    }

    for key, _ in MODELS:
        cm = np.asarray(runs[key]["metrics"]["confusion_matrix"], dtype=int)
        total_err = int(cm.sum() - np.trace(cm))
        totals[key] = total_err

        stats["Shirt -> T-shirt/top"][key] = int(cm[idx["Shirt"], idx["T-shirt/top"]])
        stats["T-shirt/top -> Shirt"][key] = int(cm[idx["T-shirt/top"], idx["Shirt"]])
        stats["Pullover <-> Coat"][key] = int(
            cm[idx["Pullover"], idx["Coat"]] + cm[idx["Coat"], idx["Pullover"]]
        )
        stats["Shirt -> Pullover / Coat"][key] = int(
            cm[idx["Shirt"], idx["Pullover"]] + cm[idx["Shirt"], idx["Coat"]]
        )

        ub_err = sum(
            int(cm[i, j]) for i in ub_indices for j in ub_indices if i != j
        )
        stats["All confusions inside {T-shirt/top, Pullover, Coat, Shirt}"][key] = ub_err

        cross_err = sum(
            int(cm[i, j] + cm[j, i]) for i in clothing_indices for j in non_clothing_indices
        )
        stats["Clothing <-> footwear / bag (cross-group)"][key] = cross_err

    lines.append(
        "| Total errors | "
        + " | ".join(f"{totals[k]:,}" for k, _ in MODELS)
        + " |"
    )
    for group_name, values in stats.items():
        cells = [
            f"{values[k]} ({100.0 * values[k] / totals[k]:.1f}%)"
            for k, _ in MODELS
        ]
        lines.append(f"| {group_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_combined_curves(runs, out_path):
    """6x2 grid: one row per model, left = train/val loss, right = train/val macro-F1."""
    import matplotlib.pyplot as plt
    from src.utils.plots import INK, MUTED, TRAIN_COLOR, VAL_COLOR, _style

    short_titles = {
        "linear": "Linear / Softmax",
        "mlp": "MLP (784-256-10)",
        "cnn": "CNN (32-64 conv-pool)",
        "lstm": "LSTM (28 rows, h=128)",
        "gru": "GRU (28 rows, h=128)",
        "transformer": "Transformer (28 rows, 2L)",
    }

    fig, axes = plt.subplots(len(MODELS), 2, figsize=(10.5, 13.5), constrained_layout=True,
                             sharex=True)
    for row_idx, (key, _) in enumerate(MODELS):
        hist = runs[key]["history"]["history"]
        best_ep = runs[key]["history"]["best_epoch"]
        epochs = [r["epoch"] for r in hist]
        title_prefix = short_titles[key]

        panels = [
            ("train_loss", "val_loss", f"{title_prefix} — Cross-entropy loss"),
            ("train_macro_f1", "val_macro_f1", f"{title_prefix} — Macro-F1 (best ep {best_ep})"),
        ]
        for col_idx, (t_key, v_key, panel_title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.plot(epochs, [r[t_key] for r in hist], color=TRAIN_COLOR,
                    linewidth=1.8, marker="o", markersize=3.2, label="train")
            ax.plot(epochs, [r[v_key] for r in hist], color=VAL_COLOR,
                    linewidth=1.8, linestyle="--", marker="s", markersize=3.2, label="validation")
            ax.axvline(best_ep, color=MUTED, linestyle=":", linewidth=1.1, alpha=0.8)
            ax.set_title(panel_title, fontsize=9.5)
            if row_idx == len(MODELS) - 1:
                ax.set_xlabel("epoch")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
            _style(ax)

    fig.suptitle("Training and validation curves across all six models (Fashion-MNIST, seed 42)",
                 fontsize=12, color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_confusion_matrices(runs, out_path):
    """3x2 grid of row-normalized test confusion matrices across all six models."""
    import matplotlib.pyplot as plt
    from src.utils.plots import INK, MUTED, SEQUENTIAL_CMAP

    short_titles = {
        "linear": "Linear / Softmax (macro-F1 0.8441)",
        "mlp": "MLP (macro-F1 0.8902)",
        "cnn": "CNN (macro-F1 0.9163)",
        "lstm": "LSTM (macro-F1 0.8898)",
        "gru": "GRU (macro-F1 0.9010)",
        "transformer": "Transformer (macro-F1 0.8825)",
    }

    classes = runs["linear"]["config"]["data"]["classes"]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 16.0), constrained_layout=True)

    for idx, (key, _) in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        cm = np.asarray(runs[key]["metrics"]["confusion_matrix"], dtype=float)
        shown = cm / cm.sum(axis=1, keepdims=True)

        image = ax.imshow(shown, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        if idx // 2 == 2:
            ax.set_xlabel("predicted", fontsize=9, color=MUTED)
        if idx % 2 == 0:
            ax.set_ylabel("true", fontsize=9, color=MUTED)
        ax.set_title(short_titles[key], fontsize=10, color=INK)

        threshold = shown.max() * 0.6
        for r in range(shown.shape[0]):
            for c in range(shown.shape[1]):
                val = shown[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if val > threshold else INK)

    bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    bar.set_label("per-class recall (row-normalized)", color=MUTED, fontsize=9)
    fig.suptitle("Row-normalized test confusion matrices across all six models (10,000 test images)",
                 fontsize=12.5, color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def regenerate_figures(runs, dataset, seed, export_assets=False):
    assets_dir = ROOT / "assets"
    for key, _ in MODELS:
        run = runs[key]
        run_dir = run["dir"]
        classes = run["config"]["data"]["classes"]
        cm = np.asarray(run["metrics"]["confusion_matrix"], dtype=float)

        curves_path = run_dir / "curves.png"
        cm_path = run_dir / "confusion_matrix_test.png"
        plot_curves(run["history"]["history"], curves_path,
                    title=f"{key} on {dataset} (seed {seed})")
        plot_confusion_matrix(cm, classes, cm_path, title=f"{key} - test split")

        if export_assets:
            assets_dir.mkdir(parents=True, exist_ok=True)
            asset_curves = assets_dir / f"curves_{key}.png"
            asset_cm = assets_dir / f"confusion_matrix_{key}.png"
            if not asset_curves.exists():
                plot_curves(run["history"]["history"], asset_curves,
                            title=f"{key} on {dataset} (seed {seed})")
            if not asset_cm.exists():
                plot_confusion_matrix(cm, classes, asset_cm,
                                      title=f"{key} - test split")

    if export_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)
        plot_combined_curves(runs, assets_dir / "curves_all_models.png")
        plot_combined_confusion_matrices(runs, assets_dir / "confusion_matrix_all_models.png")


def main():
    args = parse_args()
    runs = load_runs(args.outputs_dir, args.dataset, args.seed)

    print("### Main comparison (held-out test split)\n")
    print(format_main_table(runs))
    print("\n### Per-class F1 on the test split\n")
    print(format_per_class_table(runs))
    print("\n### Upper-body 4 vs. other 6 F1 gap\n")
    print(format_cluster_gap_table(runs))
    print("\n### Error-group breakdown\n")
    print(format_error_breakdown_table(runs))

    if args.plot or args.export_assets:
        regenerate_figures(runs, args.dataset, args.seed, export_assets=args.export_assets)


if __name__ == "__main__":
    main()
