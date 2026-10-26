"""Train one experiment from a config.

python train.py --config configs/baseline_unet.yaml
python train.py --config configs/midas_full.yaml
python train.py --config configs/midas_frozen.yaml
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import get_dataloaders
from src.engine.loss import MaskedSILogLoss, ScaleShiftInvariantLoss
from src.engine.trainer import Trainer
from src.models import build_model
from src.utils.common import count_params, env_info, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches, to check the pipeline")
    ap.add_argument("--name", default=None, help="override output folder name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name:
        cfg["name"] = args.name
        cfg["out_dir"] = os.path.join(os.path.dirname(cfg["out_dir"]), args.name)
    set_seed(cfg["seed"], cfg["deterministic"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 2 if args.smoke else cfg["train"]["epochs"]

    loaders = get_dataloaders(cfg["data"]["root"], cfg["data"]["batch_size"], cfg["data"]["num_workers"],
                              augment=cfg["data"]["augment"], seed=cfg["seed"], splits=("train", "val"))
    model, output_type = build_model(cfg)
    criterion = MaskedSILogLoss(cfg["train"].get("silog_lambda", 0.5)) if output_type == "metric" \
        else ScaleShiftInvariantLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"{cfg['name']}: {count_params(model) / 1e6:.2f}M params "
          f"({count_params(model, True) / 1e6:.2f}M trainable), device {device}, seed {cfg['seed']}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    trainer = Trainer(model, output_type, criterion, optimizer, scheduler, loaders, cfg, device,
                      max_batches=5 if args.smoke else None)
    summary = trainer.fit(epochs)
    summary.update({
        "name": cfg["name"],
        "params_M": count_params(model) / 1e6,
        "trainable_params_M": count_params(model, True) / 1e6,
        "env": env_info(),
    })
    save_json(summary, os.path.join(cfg["out_dir"], "train_summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
