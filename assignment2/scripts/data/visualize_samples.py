import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DiodeDepthDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_indoor", type=int, default=3)
    ap.add_argument("--n_outdoor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/figures/train_samples.png")
    args = ap.parse_args()

    ds = DiodeDepthDataset(args.root, args.split, normalize=False)
    rng = np.random.default_rng(args.seed)
    indoor = [i for i, s in enumerate(ds.ids) if "_indoors_" in s]
    outdoor = [i for i, s in enumerate(ds.ids) if "_outdoor_" in s]
    picks = list(rng.choice(indoor, args.n_indoor, replace=False)) + \
        list(rng.choice(outdoor, args.n_outdoor, replace=False))

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 3.3 * len(picks)))
    for row, idx in enumerate(picks):
        s = ds[idx]
        rgb = s["rgb"].permute(1, 2, 0).numpy()
        depth = s["depth"][0].numpy()
        mask = s["mask"][0].numpy()
        d = depth[mask > 0]
        # cut colour scale at p99 of this image, otherwise a few far pixels make everything one colour
        vmax = np.percentile(d, 99)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{s['id']} ({s['domain']})", fontsize=9)

        im = axes[row, 1].imshow(np.ma.masked_where(mask == 0, depth), cmap=cmap, vmin=0, vmax=vmax)
        axes[row, 1].set_title(f"depth (colour cut at p99 = {vmax:.1f} m, max {d.max():.1f} m)", fontsize=9)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.035, label="m")

        axes[row, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"mask (white = valid), valid {mask.mean() * 100:.1f}%", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")
        print(f"{s['id']:32s} {s['domain']:8s} valid={mask.mean() * 100:5.1f}%  "
              f"median={np.median(d):5.1f} m  max={d.max():6.1f} m")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
