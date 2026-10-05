import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import SPLITS, DiodeDepthDataset, get_dataloaders


def describe(sample):
    rgb, depth, mask = sample["rgb"], sample["depth"], sample["mask"]
    valid = mask.bool()
    d = depth[valid]
    print(f"  id      : {sample['id']} ({sample['domain']})")
    print(f"  rgb     : shape={tuple(rgb.shape)} dtype={rgb.dtype} range=[{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"  depth   : shape={tuple(depth.shape)} dtype={depth.dtype} "
          f"valid range=[{d.min():.3f}, {d.max():.3f}] m, mean={d.mean():.3f} m, "
          f"NaN={torch.isnan(depth).sum().item()}")
    print(f"  mask    : shape={tuple(mask.shape)} unique={mask.unique().tolist()} "
          f"valid={100 * valid.float().mean():.2f}%")
    # PASS if size matches, depth is ok, mask has both 0 and 1
    same_hw = rgb.shape[1:] == depth.shape[1:] == mask.shape[1:]
    ok = same_hw and d.numel() > 0 and 0 < valid.float().mean() < 1 and not torch.isnan(depth).any()
    print(f"  checks  : same HxW={same_hw}, depth>0 present={d.numel() > 0}, "
          f"mask has 0 and 1={0 < valid.float().mean() < 1} -> {'PASS' if ok else 'CHECK'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    args = ap.parse_args()

    for split in SPLITS:
        ds = DiodeDepthDataset(args.root, split)
        print(f"[{split}] {len(ds)} samples")

    # test 1 indoor and 1 outdoor sample from train
    ds = DiodeDepthDataset(args.root, "train")
    first_out = next(i for i, s in enumerate(ds.ids) if "_outdoor_" in s)
    for name, idx in [("train indoor", 0), ("train outdoor", first_out)]:
        print(f"\n== {name} sample (index {idx}) ==")
        describe(ds[idx])

    batch = next(iter(get_dataloaders(args.root, batch_size=4, num_workers=0)["train"]))
    print(f"\nbatch: rgb={tuple(batch['rgb'].shape)} depth={tuple(batch['depth'].shape)} "
          f"mask={tuple(batch['mask'].shape)}")


if __name__ == "__main__":
    main()
