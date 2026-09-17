"""Evaluate a trained run on the held-out test split.

Writes test metrics, a confusion matrix, and correct/incorrect prediction examples
into the run directory produced by scripts/train.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import get_dataloaders
from src.engine.trainer import measure_inference_time, run_epoch
from src.utils.metrics import count_parameters, per_class_f1
from src.utils.plots import plot_confusion_matrix, plot_sample_grid
from src.utils.seed import set_seed

from scripts.train import build_model  # noqa: E402  (same model registry as training)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="directory written by scripts/train.py, e.g. outputs/linear_fashion_mnist_seed42")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-examples", type=int, default=16)
    return parser.parse_args()


def collect_examples(loader, y_true, y_pred, normalize, limit):
    """Pull `limit` correct and `limit` incorrect images back out of the loader, denormalized."""
    images = torch.cat([batch for batch, _ in loader]).numpy()
    images = images * normalize["std"] + normalize["mean"]

    correct = np.flatnonzero(y_true == y_pred)[:limit]
    wrong = np.flatnonzero(y_true != y_pred)[:limit]
    return (images[correct], correct), (images[wrong], wrong)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    config = run_config["config"]
    data_info = run_config["data"]
    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, _ = get_dataloaders(seed=seed, **config["data"])
    loader = loaders[args.split]

    model = build_model(config["model"], data_info["input_shape"],
                        data_info["num_classes"]).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics, y_true, y_pred = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    timing = measure_inference_time(model, loader, device)
    class_names = data_info["classes"]

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix(matrix, class_names, run_dir / f"confusion_matrix_{args.split}.png",
                          title=f"{config['model']['name']} - {args.split} split")

    (correct_images, correct_idx), (wrong_images, wrong_idx) = collect_examples(
        loader, y_true, y_pred, data_info["normalize"], args.num_examples)
    plot_sample_grid(correct_images, y_true[correct_idx], class_names,
                     run_dir / f"examples_correct_{args.split}.png",
                     title="Correct predictions")
    plot_sample_grid(wrong_images, y_true[wrong_idx], class_names,
                     run_dir / f"examples_incorrect_{args.split}.png",
                     pred_labels=y_pred[wrong_idx], title="Incorrect predictions (true -> predicted)")

    report = {
        "run_name": run_config["run_name"],
        "model": config["model"]["name"],
        "split": args.split,
        "checkpoint_epoch": checkpoint["epoch"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "loss": metrics["loss"],
        "per_class_f1": dict(zip(class_names,
                                 per_class_f1(y_true, y_pred, len(class_names)))),
        "parameters": count_parameters(model),
        "inference": timing,
        "confusion_matrix": matrix.tolist(),
    }
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")

    print(f"{config['model']['name']} on {args.split}: "
          f"accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f}")
    print(f"parameters: {report['parameters']['total']:,}")
    print(f"inference : {timing['ms_per_image']:.4f} ms/image "
          f"({timing['total_seconds']:.2f}s for {timing['num_images']} images)")
    print(f"artifacts : {run_dir}")


if __name__ == "__main__":
    main()
