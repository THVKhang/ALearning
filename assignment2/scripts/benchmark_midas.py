# Time MiDaS small on our data for the compute estimate (no weights saved, no metrics).

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DiodeDepthDataset


def sync():
    torch.cuda.synchronize()


def gb(x):
    return x / 1024 ** 3


def time_inference(model, ds, idx, batch_size):
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, num_workers=4)
    batches = [b["rgb"] for b in loader]  # load first so disk time is not counted
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for x in batches[:3]:  # warm up
            model(x.cuda())
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for x in batches:
            model(x.cuda())
        sync()
    sec = time.perf_counter() - t0
    return sec / len(idx), gb(torch.cuda.max_memory_allocated()), gb(torch.cuda.max_memory_reserved())


def time_training(model, ds, batch_size, iters, amp):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                        generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler(enabled=amp)
    model.train()
    torch.cuda.empty_cache()
    it = iter(loader)
    warmup = 5
    for i in range(warmup + iters):
        if i == warmup:
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        b = next(it)
        x, depth, mask = b["rgb"].cuda(), b["depth"].cuda(), b["mask"].cuda()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(x).unsqueeze(1)
        # simple masked L1 on inverse depth, just to get real backward cost (real loss is picked later)
        target = torch.where(mask > 0, 1.0 / depth.clamp(min=1e-3), torch.zeros_like(depth))
        loss = ((pred.float() - target).abs() * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    sync()
    sec = time.perf_counter() - t0
    return sec / iters, gb(torch.cuda.max_memory_allocated()), gb(torch.cuda.max_memory_reserved())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    ap.add_argument("--n_images", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--train_iters", type=int, default=40)
    ap.add_argument("--out", default="outputs/benchmark/midas_small.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    torch.backends.cudnn.benchmark = True

    # MiDaS_small pulls its backbone from this repo, trust it first or torch.hub waits for a y/N input
    torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_lite3", pretrained=False, trust_repo=True)
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).cuda()
    n_params = sum(p.numel() for p in model.parameters())

    train = DiodeDepthDataset(args.root, "train")
    n_train, n_val = len(train), len(DiodeDepthDataset(args.root, "val"))
    idx = np.random.default_rng(42).choice(n_train, args.n_images, replace=False).tolist()

    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
           "model": "MiDaS_small (intel-isl/MiDaS, torch.hub)", "params_M": n_params / 1e6,
           "input": "3x288x384", "n_images": args.n_images, "batch_size": args.batch_size}

    res["infer_bs1_s_per_img"], res["infer_bs1_alloc_GB"], res["infer_bs1_reserved_GB"] = \
        time_inference(model, train, idx, 1)
    res["infer_bs8_s_per_img"], res["infer_bs8_alloc_GB"], res["infer_bs8_reserved_GB"] = \
        time_inference(model, train, idx, args.batch_size)

    iters_per_epoch = math.ceil(n_train / args.batch_size)
    val_sec = n_val * res["infer_bs8_s_per_img"]
    for amp in (False, True):
        tag = "amp" if amp else "fp32"
        s_it, alloc, reserved = time_training(model, train, args.batch_size, args.train_iters, amp)
        epoch_min = (iters_per_epoch * s_it + val_sec) / 60
        res[f"train_{tag}"] = {"s_per_iter": s_it, "alloc_GB": alloc, "reserved_GB": reserved,
                               "epoch_min": epoch_min, "10_epochs_h": 10 * epoch_min / 60,
                               "20_epochs_h": 20 * epoch_min / 60}

    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
