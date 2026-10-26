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
            plot_curves(run["history"]["history"], assets_dir / f"curves_{key}.png",
                        title=f"{key} on {dataset} (seed {seed})")
            plot_confusion_matrix(cm, classes, assets_dir / f"confusion_matrix_{key}.png",
                                  title=f"{key} - test split")


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
