"""Per-image test metrics for every seed-42 model, plus a qualitative figure of the
best / median / worst test images (ranked by MiDaS full FT Abs Rel) in each domain.

python scripts/report/per_image_test.py
Writes outputs/diagnostics/per_image_test.csv and assets/a2_qualitative_test.png.
"""

import csv
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
from src.engine.metrics import depth_metrics, valid_mask
from src.models import build_model

MODELS = [("U-Net (aligned)", "baseline_unet", "unet", False),
          ("MiDaS zero-shot", "midas_zeroshot", "midas_small", False),
          ("MiDaS full FT", "midas_full", "midas_small", False),
          ("MiDaS frozen enc.", "midas_frozen", "midas_small", True)]
RANK_BY = "midas_full"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(run, model_name, freeze):
    model, kind = build_model({"model": model_name, "freeze_encoder": freeze})
    if run != "midas_zeroshot":
        ckpt = os.path.join(ROOT, "outputs", run, "best.pth")
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return model.to(DEVICE).eval(), kind


@torch.no_grad()
def predict(model, kind, s):
    out = model(s["rgb"][None].to(DEVICE)).float()[0, 0].cpu()
    disp = 1.0 / out.clamp(min=1e-3) if kind == "metric" else out
    return align_disparity(disp, s["depth"][0], s["mask"][0], MAX_DEPTH[s["domain"]])[0]


@torch.no_grad()
def main():
    torch.manual_seed(42)
    ds = DiodeDepthDataset(os.path.join(ROOT, "data/diode_subset"), "test")
    models = [(label, run, *load(run, name, fr)) for label, run, name, fr in MODELS]

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        gt, mask, max_d = s["depth"][0], s["mask"][0], MAX_DEPTH[s["domain"]]
        valid = valid_mask(gt, mask, max_d)
        row = {"idx": i, "id": s["id"], "domain": s["domain"],
               "valid_frac": float(valid.float().mean()),
               "gt_median": float(gt[valid].median()) if valid.any() else float("nan")}
        for _, run, model, kind in models:
            m = depth_metrics(predict(model, kind, s), gt, mask, max_d)
            for k in ("abs_rel", "rmse", "delta1"):
                row[f"{run}_{k}"] = m[k] if m else float("nan")
        rows.append(row)

    out_dir = os.path.join(ROOT, "outputs", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_image_test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved per-image metrics for", len(rows), "test images")

    # best / median / worst by the ranking model, per domain
    picks = []
    for dom in ("indoor", "outdoor"):
        r = sorted((x for x in rows if x["domain"] == dom), key=lambda x: x[f"{RANK_BY}_abs_rel"])
        picks += [("best", r[0]), ("median", r[len(r) // 2]), ("worst", r[-1])]

    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    cols = 2 + len(models)
    fig, axes = plt.subplots(len(picks), cols + 1, figsize=(2.6 * cols + 0.4, 2.05 * len(picks)),
                             gridspec_kw={"width_ratios": [1] * cols + [0.05]})
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d9d9d6")
    for r, (tag, row) in enumerate(picks):
        s = ds[row["idx"]]
        gt, mask = s["depth"][0], s["mask"][0]
        valid = valid_mask(gt, mask, MAX_DEPTH[s["domain"]])
        vmin, vmax = np.percentile(gt[valid].numpy(), [2, 98])
        rgb = (s["rgb"] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        panels = [("RGB", rgb, None), ("Ground truth", np.where(valid, gt, np.nan), None)]
        for label, run, model, kind in models:
            panels.append((label, predict(model, kind, s).numpy(), f"AbsRel {row[f'{run}_abs_rel']:.3f}"))
        for c, (title, img, note) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img) if c == 0 else ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if note:
                ax.text(5, 22, note, color="white", fontsize=7, bbox=dict(facecolor="black", alpha=0.6, lw=0))
            if c == 0:
                ax.set_ylabel(f"{s['domain']} · {tag}\nvalid {100 * row['valid_frac']:.0f}%", fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, cax=axes[r, -1]).set_label("depth (m)", fontsize=7)
    fig.tight_layout()
    path = os.path.join(ROOT, "assets", "a2_qualitative_test.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
