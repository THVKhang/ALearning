"""Train one multimodal or unimodal experiment from a config.

Usage:
    python train.py --config configs/rgb_only.yaml
    python train.py --config configs/depth_only.yaml
    python train.py --config configs/simple_fusion.yaml
    python train.py --config configs/improved_fusion.yaml
    python train.py --config configs/ablation_no_gate.yaml
    python train.py --config configs/ablation_no_aux.yaml
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import get_dataloaders
from src.engine.loss import MultimodalJointLoss
from src.engine.trainer import Trainer
from src.models.multimodal import build_model
from src.utils.common import count_params, env_info, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override random seed")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches sanity check")
    ap.add_argument("--name", default=None, help="override output folder name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        if args.name is None and args.seed != 42:
            cfg["name"] = f"{cfg['name']}_seed{args.seed}"
            cfg["out_dir"] = os.path.join(os.path.dirname(cfg["out_dir"]), cfg["name"])
    if args.name:
        cfg["name"] = args.name
        cfg["out_dir"] = os.path.join(os.path.dirname(cfg["out_dir"]), args.name)

    set_seed(cfg["seed"], cfg["deterministic"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 2 if args.smoke else cfg["train"]["epochs"]

    loaders = get_dataloaders(
        cfg["data"]["root"],
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
        seed=cfg["seed"],
        splits=("train", "val"),
    )
    model = build_model(cfg)
    criterion = MultimodalJointLoss(aux_weight=cfg["train"].get("aux_weight", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(
        f"{cfg['name']}: {count_params(model):,} params "
        f"({count_params(model, True):,} trainable), device={device}, seed={cfg['seed']}"
    )
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        loaders,
        cfg,
        device=device,
        max_batches=5 if args.smoke else None,
    )
    summary = trainer.fit(epochs)
    summary.update(
        {
            "name": cfg["name"],
            "params": count_params(model),
            "trainable_params": count_params(model, True),
            "params_M": count_params(model) / 1e6,
            "trainable_params_M": count_params(model, True) / 1e6,
            "env": env_info(),
        }
    )
    save_json(summary, os.path.join(cfg["out_dir"], "train_summary.json"))


if __name__ == "__main__":
    main()
