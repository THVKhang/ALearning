"""Side-by-side val predictions: RGB | GT | U-Net | MiDaS zero-shot | MiDaS full FT.
Each row uses one colour scale (from its GT) so the columns are comparable.

python scripts/report/qualitative.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, MAX_DEPTH, DiodeDepthDataset
from src.engine.align import align_disparity
from src.engine.metrics import depth_metrics
from src.models import build_model

MODELS = [("U-Net baseline", "baseline_unet", "unet", False),
          ("MiDaS zero-shot", "midas_zeroshot", "midas_small", False),
          ("MiDaS full FT", "midas_full", "midas_small", False),
          ("MiDaS frozen", "midas_frozen", "midas_small", True)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(run, model_name, freeze):
    model, kind = build_model({"model": model_name, "freeze_encoder": freeze})
    ckpt = os.path.join(ROOT, "outputs", run, "best.pth")
    if run != "midas_zeroshot":
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return model.to(DEVICE).eval(), kind


def pick(ds, n_per_domain):
    ind = [i for i, s in enumerate(ds.ids) if "_indoors_" in s]
    out = [i for i, s in enumerate(ds.ids) if "_indoors_" not in s]
    take = lambda idx: [idx[int(k)] for k in np.linspace(0, len(idx) - 1, n_per_domain + 2)[1:-1]]
    return take(ind) + take(out)


@torch.no_grad()
def main():
    ds = DiodeDepthDataset(os.path.join(ROOT, "data/diode_subset"), "val")
    idx = pick(ds, 3)
    models = [(label, *load(run, name, fr)) for label, run, name, fr in MODELS]

    cols = 2 + len(models)
    fig, axes = plt.subplots(len(idx), cols + 1, figsize=(3 * cols + 0.4, 2.4 * len(idx)),
                             gridspec_kw={"width_ratios": [1] * cols + [0.05]})
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for r, i in enumerate(idx):
        s = ds[i]
        gt, mask, max_d = s["depth"][0], s["mask"][0], MAX_DEPTH[s["domain"]]
        valid = mask > 0
        vmin, vmax = np.percentile(gt[valid].numpy(), [2, 98])
        rgb = (s["rgb"] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        panels = [("RGB", rgb, None), ("Ground truth", np.where(valid, gt, np.nan), None)]
        for label, model, kind in models:
            out = model(s["rgb"][None].to(DEVICE)).float()[0, 0].cpu()
            pred = out if kind == "metric" else align_disparity(out, gt, mask, max_d)[0]
            m = depth_metrics(pred, gt, mask, max_d)
            panels.append((label, pred.numpy(), f"AbsRel {m['abs_rel']:.3f}"))

        for c, (title, img, note) in enumerate(panels):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
            else:
                cmap = plt.get_cmap("magma_r").copy()
                cmap.set_bad("lightgray")
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title if r == 0 else "", fontsize=9)
            if note:
                ax.text(4, 20, note, color="white", fontsize=7, bbox=dict(facecolor="black", alpha=0.6))
            if c == 0:
                ax.set_ylabel(f"{s['domain']}\n{s['id'][:11]}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, cax=axes[r, -1], label="m")

    fig.tight_layout()
    path = os.path.join(ROOT, "results", "qualitative.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
