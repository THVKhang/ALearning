"""Train one Assignment-1 model; writes checkpoint, history and curves to outputs/.

Test data is never touched here - use scripts/eval.py for the held-out test split.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `python scripts/train.py` work, not only `-m scripts.train`
    sys.path.insert(0, str(ROOT))

from src.data.dataset import get_dataloaders
from src.engine.trainer import fit
from src.models.linear import LinearClassifier
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.utils.metrics import count_parameters
from src.utils.plots import plot_curves
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model", choices=["linear", "mlp","cnn"])
    parser.add_argument("--dataset", choices=["fashion_mnist", "mnist"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    return parser.parse_args()


def load_config(args):
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    overrides = {
        ("model", "name"): args.model,
        ("data", "dataset"): args.dataset,
        ("data", "batch_size"): args.batch_size,
        ("train", "epochs"): args.epochs,
        ("train", "lr"): args.lr,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.seed is not None:
        config["seed"] = args.seed

    # Resolve data_root against the project root so the dataset is found and not
    # re-downloaded when the script is launched from another working directory.
    data_root = Path(config["data"]["data_root"])
    if not data_root.is_absolute():
        config["data"]["data_root"] = str(ROOT / data_root)
    return config


def build_model(model_config, input_shape, num_classes):
    name = model_config["name"]
    if name == "linear":
        return LinearClassifier(input_shape, num_classes)
    if name == "mlp":
        return MLP(input_shape, tuple(model_config["hidden_sizes"]), num_classes,
                   model_config.get("dropout", 0.0))
    if name == "cnn":
        return CNN(input_shape, tuple(model_config["channels"]),
                   model_config["classifier_hidden"], num_classes,
                   model_config.get("dropout", 0.0))
    raise ValueError(f"unknown model: {name}")


def environment_info(device):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def main():
    args = parse_args()
    config = load_config(args)
    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, data_info = get_dataloaders(seed=seed, **config["data"])

    model = build_model(config["model"], data_info["input_shape"],
                        data_info["num_classes"]).to(device)
    parameters = count_parameters(model)

    run_name = args.run_name or f"{config['model']['name']}_{data_info['dataset']}_seed{seed}"
    run_dir = Path(args.outputs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run        : {run_name}")
    print(f"device     : {device}")
    print(f"model      : {config['model']['name']} ({parameters['total']:,} parameters)")
    print(f"split sizes: {data_info['split_sizes']}")

    summary = fit(
        model, loaders["train"], loaders["val"], device,
        checkpoint_path=run_dir / "best.pt", **config["train"],
    )

    plot_curves(summary["history"], run_dir / "curves.png",
                title=f"{config['model']['name']} on {data_info['dataset']} (seed {seed})")

    (run_dir / "config.json").write_text(json.dumps({
        "run_name": run_name,
        "config": config,
        "data": data_info,
        "parameters": parameters,
        "environment": environment_info(device),
    }, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"best epoch : {summary['best_epoch']} "
          f"(val {summary['monitor']} {summary['best_val_score']:.4f})")
    print(f"train time : {summary['train_seconds_total']:.1f}s total, "
          f"{summary['train_seconds_per_epoch']:.1f}s/epoch")
    print(f"artifacts  : {run_dir}")


if __name__ == "__main__":
    main()
