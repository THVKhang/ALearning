"""Evaluate a trained unimodal or multimodal model on val or test, including missing-modality stress tests.

Usage:
    python evaluate.py --config configs/improved_fusion.yaml --split test
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import get_dataloaders
from src.models.multimodal import build_model
from src.utils.common import count_params, env_info, load_config, save_json, set_seed
from src.utils.metrics import compute_classification_metrics


def load_model(cfg, ckpt_path, device):
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), int(state.get("epoch", 0))


@torch.no_grad()
def measure_latency_ms(model, dataset, device, amp, n=200):
    loader = DataLoader(Subset(dataset, range(min(n, len(dataset)))), batch_size=1)
    batches = [
        {
            "rgb": b["rgb"].to(device),
            "depth_geo": b["depth_geo"].to(device),
            "valid_ratio": b["valid_ratio"].to(device),
        }
        for b in loader
    ]
    is_cuda = str(device).startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp and is_cuda):
        for b in batches[:10]:
            model(b)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for b in batches:
            model(b)
        if is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(batches) * 1000.0


@torch.no_grad()
def run_eval(model, loader, device, amp, drop_modality=None):
    is_cuda = str(device).startswith("cuda")
    y_true, y_pred, probs_max, ids, valid_ratios, gates = [], [], [], [], [], []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth_geo = batch["depth_geo"].to(device, non_blocking=True)
        v_ratio = batch["valid_ratio"].to(device, non_blocking=True)

        if drop_modality == "depth":
            depth_geo = torch.zeros_like(depth_geo)
            v_ratio = torch.zeros_like(v_ratio)
        elif drop_modality == "rgb":
            rgb = torch.zeros_like(rgb)

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and is_cuda):
            out = model({"rgb": rgb, "depth_geo": depth_geo, "valid_ratio": v_ratio})
        logits = out["logits"].float()
        prob = torch.softmax(logits, dim=-1)
        conf, pred = prob.max(dim=-1)

        y_true.extend(batch["label"].tolist())
        y_pred.extend(pred.cpu().tolist())
        probs_max.extend(conf.cpu().tolist())
        ids.extend(batch["id"])
        valid_ratios.extend(batch["valid_ratio"].squeeze(-1).tolist())
        if "gate_mean" in out:
            gates.extend(out["gate_mean"].float().cpu().tolist())

    res = compute_classification_metrics(y_true, y_pred)

    # Subset analysis by LiDAR missing/invalid pixel rate (high dropout > 35% vs clean <= 35%)
    vr = np.asarray(valid_ratios)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    high_drop = vr < 0.65
    low_drop = vr >= 0.65
    res["by_validity_regime"] = {
        "clean_lidar_ge_65pct_valid": (
            compute_classification_metrics(yt[low_drop], yp[low_drop])["macro_f1"] if low_drop.any() else None
        ),
        "noisy_lidar_lt_65pct_valid": (
            compute_classification_metrics(yt[high_drop], yp[high_drop])["macro_f1"] if high_drop.any() else None
        ),
        "n_clean": int(low_drop.sum()),
        "n_noisy": int(high_drop.sum()),
    }
    if gates:
        g_arr = np.asarray(gates)
        res["gate_rgb_weight_clean"] = float(g_arr[low_drop].mean()) if low_drop.any() else None
        res["gate_rgb_weight_noisy"] = float(g_arr[high_drop].mean()) if high_drop.any() else None

    records = {
        "ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": [round(float(c), 4) for c in probs_max],
        "valid_ratio": [round(float(v), 4) for v in valid_ratios],
    }
    return res, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
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
    is_cuda = device == "cuda"
    amp = cfg["train"].get("amp", True) and is_cuda
    ckpt = args.ckpt or os.path.join(cfg["out_dir"], "best.pth")
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")

    model, ckpt_epoch = load_model(cfg, ckpt, device)
    loader = get_dataloaders(
        cfg["data"]["root"],
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["eval"]["batch_size"],
        num_workers=0,
        augment=False,
        splits=(args.split,),
    )[args.split]

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res, records = run_eval(model, loader, device, amp)
    if is_cuda:
        torch.cuda.synchronize()
    res["eval_time_s"] = round(time.perf_counter() - t0, 3)
    res["infer_ms_per_sample"] = round(measure_latency_ms(model, loader.dataset, device, amp), 4)
    res["checkpoint_epoch"] = ckpt_epoch
    res["name"] = cfg["name"]
    res["split"] = args.split
    res["params"] = count_params(model)
    res["params_M"] = round(count_params(model) / 1e6, 4)

    # Missing-modality stress evaluation for multimodal models (handbook Section 29.2)
    if cfg["model"] in ("simple_fusion", "improved_fusion"):
        res_no_depth, _ = run_eval(model, loader, device, amp, drop_modality="depth")
        res_no_rgb, _ = run_eval(model, loader, device, amp, drop_modality="rgb")
        res["missing_modality_stress"] = {
            "full_multimodal_macro_f1": res["macro_f1"],
            "drop_depth_at_test_macro_f1": res_no_depth["macro_f1"],
            "drop_rgb_at_test_macro_f1": res_no_rgb["macro_f1"],
            "drop_depth_at_test_acc": res_no_depth["accuracy"],
            "drop_rgb_at_test_acc": res_no_rgb["accuracy"],
        }

    res["env"] = env_info()
    save_json(res, os.path.join(cfg["out_dir"], f"eval_{args.split}.json"))
    save_json(records, os.path.join(cfg["out_dir"], f"predictions_{args.split}.json"))
    print(
        f"{cfg['name']:18s} [{args.split:4s}] Acc={res['accuracy']:.4f} MacroF1={res['macro_f1']:.4f} "
        f"(Dom={res['domain_accuracy']:.4f}, Elev={res['elevation_accuracy']:.4f}) "
        f"| {res['infer_ms_per_sample']:.3f} ms/pair"
    )


if __name__ == "__main__":
    main()
