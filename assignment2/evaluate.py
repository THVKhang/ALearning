"""Evaluate a model on val (or test).

python evaluate.py --config configs/midas_zeroshot.yaml            # no checkpoint
python evaluate.py --config configs/baseline_unet.yaml             # uses outputs/baseline_unet/best.pth
python evaluate.py --config configs/midas_full.yaml --split test

Metric models (U-Net) are scored twice: raw metres, and with the same per-image
scale/shift alignment as MiDaS so the comparison is fair.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import get_dataloaders
from src.engine.align import align_batch, depth_to_disp
from src.engine.metrics import MetricTracker
from src.models import build_model
from src.utils.common import count_params, env_info, load_config, save_json, set_seed


def load_model(cfg, ckpt, device):
    model, output_type = build_model(cfg)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded {ckpt} (epoch {state['epoch']})")
    return model.to(device).eval(), output_type


def default_ckpt(cfg):
    path = os.path.join(cfg["out_dir"], "best.pth")
    return path if os.path.exists(path) else None


@torch.no_grad()
def time_per_image(model, dataset, device, amp, n=100):
    loader = DataLoader(Subset(dataset, range(min(n, len(dataset)))), batch_size=1)
    xs = [b["rgb"].to(device) for b in loader]
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for x in xs[:10]:
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for x in xs:
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(xs) * 1000


@torch.no_grad()
def evaluate(model, output_type, loader, device, amp):
    trackers = {"aligned": MetricTracker()}
    if output_type == "metric":
        trackers["raw"] = MetricTracker()
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        dom = batch["domain"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(rgb)
        out = out.float()
        if output_type == "metric":
            trackers["raw"].update(out, gt, mask, dom)
            out = depth_to_disp(out)
        trackers["aligned"].update(align_batch(out, gt, mask, dom), gt, mask, dom)
    return {k: t.summary() for k, t in trackers.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["deterministic"])
    device = "cuda"
    amp = cfg["train"]["amp"]
    ckpt = args.ckpt or default_ckpt(cfg)
    if cfg["name"] != "midas_zeroshot" and ckpt is None:
        sys.exit(f"no checkpoint for {cfg['name']}, train it first")

    model, output_type = load_model(cfg, ckpt, device)
    loader = get_dataloaders(cfg["data"]["root"], cfg["eval"]["batch_size"], cfg["data"]["num_workers"],
                             splits=(args.split,))[args.split]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = evaluate(model, output_type, loader, device, amp)
    torch.cuda.synchronize()
    res["eval_time_s"] = time.perf_counter() - t0
    res["infer_ms_per_image"] = time_per_image(model, loader.dataset, device, amp)
    res.update({"name": cfg["name"], "split": args.split, "checkpoint": ckpt, "output_type": output_type,
                "params_M": count_params(model) / 1e6, "env": env_info()})

    for k in ("raw", "aligned"):
        if k in res:
            for dom in ("overall", "indoor", "outdoor"):
                m = res[k][dom]
                print(f"{cfg['name']:15s} {k:7s} {dom:8s} AbsRel {m['abs_rel']:.4f} "
                      f"RMSE {m['rmse']:.3f} d1 {m['delta1']:.3f} (n={m['n_images']})")
    print(f"eval {res['eval_time_s']:.1f}s, {res['infer_ms_per_image']:.2f} ms/img (bs1)")
    save_json(res, os.path.join(cfg["out_dir"], f"eval_{args.split}.json"))


if __name__ == "__main__":
    main()
