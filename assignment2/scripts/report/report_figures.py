"""Figures for the A2 final report (all saved into assets/, a2_*.png).

python scripts/report/report_figures.py

Reads only saved run artefacts (eval_*.json, history.csv, train_summary.json,
outputs/diagnostics/*), except the example-batch figure which loads one train batch.
Run scripts/report/per_image_test.py and edge_and_alignment_check.py first.
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")
ASSETS = os.path.join(ROOT, "assets")

# categorical slots 1-4 of the reference palette, fixed per model
MODELS = [("U-Net baseline", "baseline_unet", "#2a78d6"),
          ("MiDaS zero-shot", "midas_zeroshot", "#eb6834"),
          ("MiDaS full FT", "midas_full", "#1baf7a"),
          ("MiDaS frozen enc.", "midas_frozen", "#eda100")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def load(run, name):
    p = os.path.join(OUT, run, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def bar_labels(ax, bars, fmt="{:.3f}", size=7):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom",
                    fontsize=size, color=INK)


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    stages = [
        ("Raw data", "DIODE RGB 1024×768\nlaser depth + mask"),
        ("Preprocessing", "resize 384×288\n(RGB bilinear, depth NN)\nmask 0.6–50/200 m"),
        ("Data loader", "scene-level split\nbatch 8, hflip p=0.5\nImageNet norm."),
        ("Model", "U-Net (scratch)\nMiDaS_small\n(full / frozen enc.)"),
        ("Loss", "masked SILog (U-Net)\nmasked SSI (MiDaS)"),
        ("Optimization", "AdamW + cosine\nAMP, seed 42/43/44\nbest val Abs Rel"),
        ("Prediction", "log-depth (m)\nor relative\ninverse depth"),
        ("Post-processing", "per-image LS\nscale+shift in\ndisparity space"),
        ("Evaluation", "Abs Rel, RMSE, δ1\nindoor / outdoor\nper distance bin"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ax.grid(False)
    w, h, gap = 1.3, 2.1, 0.15
    x0 = (13 - (len(stages) * w + (len(stages) - 1) * gap)) / 2
    groups = {0: "#dbe9f8", 1: "#dbe9f8", 2: "#dbe9f8", 3: "#d6f1e6", 4: "#d6f1e6", 5: "#d6f1e6",
              6: "#fbe3d8", 7: "#fbe3d8", 8: "#fbe3d8"}
    for i, (title, body) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.55), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=groups[i], ec=INK2, lw=0.8))
        ax.text(x + w / 2, 0.55 + h - 0.22, title, ha="center", va="top", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + w / 2, 0.55 + h / 2 - 0.2, body, ha="center", va="center", fontsize=6.6, color=INK2,
                linespacing=1.35)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 0.55 + h / 2), (x + w + gap, 0.55 + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))
    for (a, b, label) in ((0, 2, "data (Part 1)"), (3, 5, "learning (Part 2)"), (6, 8, "inference & scoring")):
        xa, xb = x0 + a * (w + gap), x0 + b * (w + gap) + w
        ax.plot([xa, xb], [0.35, 0.35], color=INK2, lw=0.8)
        ax.text((xa + xb) / 2, 0.12, label, ha="center", fontsize=7.5, color=INK2, style="italic")
    ax.text(6.5, 3.1, "Monocular depth estimation pipeline (DIODE subset, 7,195 pairs)", ha="center",
            fontsize=10, weight="bold", color=INK)
    save(fig, "a2_pipeline.png")


# ---------------------------------------------------------------- example batch
def fig_example_batch():
    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
    torch.manual_seed(42)
    loader = get_dataloaders(os.path.join(ROOT, "data/diode_subset"), batch_size=8, num_workers=0,
                             augment=True, seed=42, splits=("train",))["train"]
    b = next(iter(loader))
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    order = [i for i in range(8) if b["domain"][i] == "indoor"][:2] + \
            [i for i in range(8) if b["domain"][i] == "outdoor"][:2]
    order = (order + [i for i in range(8) if i not in order])[:4]
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.4))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d9d9d6")
    for c, i in enumerate(order):
        rgb = (b["rgb"][i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        d, m = b["depth"][i, 0].numpy(), b["mask"][i, 0].numpy()
        axes[0, c].imshow(rgb)
        axes[0, c].set_title(f"{b['domain'][i]} · {b['id'][i][:11]}", fontsize=8, weight="normal")
        dv = np.where(m > 0, d, np.nan)
        vmin, vmax = np.nanpercentile(dv, [2, 98])
        im = axes[1, c].imshow(dv, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[1, c], fraction=0.035, pad=0.02).set_label("m", fontsize=7)
        axes[2, c].imshow(m, cmap="gray", vmin=0, vmax=1)
        axes[2, c].set_xlabel(f"valid {100 * m.mean():.1f}%", fontsize=8)
        for r in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
    x = b["rgb"]
    axes[0, 0].set_ylabel(f"rgb\n{tuple(x.shape)} float32\nmean {x.mean():.2f}, std {x.std():.2f}", fontsize=7.5)
    axes[1, 0].set_ylabel(f"depth (m)\n{tuple(b['depth'].shape)} float32", fontsize=7.5)
    axes[2, 0].set_ylabel(f"mask\n{tuple(b['mask'].shape)} {{0,1}}", fontsize=7.5)
    fig.suptitle("One training batch after preprocessing (batch size 8, 4 shown; hflip applied at random)",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "a2_example_batch.png")


# ---------------------------------------------------------------- curves
def read_hist(run):
    p = os.path.join(OUT, run, "history.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        h = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in h]) for k in h[0]}


def fig_curves():
    trained = [m for m in MODELS if m[1] != "midas_zeroshot"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    axes = axes.ravel()
    for ax, (label, run, color) in zip(axes[:3], trained):
        h = read_hist(run)
        ax.plot(h["epoch"], h["train_loss"], color=INK2, label="train loss")
        ax.plot(h["epoch"], h["val_loss"], color=color, label="val loss")
        best = int(np.argmin(h["overall_abs_rel"]))
        ax.axvline(h["epoch"][best], color=INK2, ls=":", lw=1)
        ax.text(h["epoch"][best], ax.get_ylim()[1], f" best ep {int(h['epoch'][best])}", fontsize=7,
                color=INK2, va="top")
        ax.set_title(f"{label} — {'SILog' if run == 'baseline_unet' else 'SSI'} loss")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7.5)
    ax = axes[3]
    h = read_hist("baseline_unet")
    ax.plot(h["epoch"], h["overall_abs_rel"], color=MODELS[0][2], marker="o", ms=3)
    ax.set_title("U-Net val Abs Rel (raw metres)")
    ax.set_xlabel("epoch")
    ax = axes[4]
    for label, run, color in trained[1:]:
        h = read_hist(run)
        ax.plot(h["epoch"], h["overall_abs_rel"], color=color, label=label, marker="o", ms=4)
    ax.set_title("MiDaS val Abs Rel (aligned)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7.5)
    axes[5].axis("off")
    axes[5].text(0, 0.5, "Checkpoint rule: epoch with the lowest\noverall val Abs Rel (dotted line).\n\n"
                 "Loss scales differ (SILog vs SSI),\nso each model has its own panel.",
                 fontsize=8.5, color=INK2, va="center")
    fig.tight_layout()
    save(fig, "a2_curves.png")


# ---------------------------------------------------------------- main results
def fig_results():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharey="row")
    doms = ["overall", "indoor", "outdoor"]
    w = 0.2
    for c, split in enumerate(("val", "test")):
        for r, (metric, nice) in enumerate((("abs_rel", "Abs Rel ↓"), ("delta1", "δ1 ↑"))):
            ax = axes[r, c]
            for i, (label, run, color) in enumerate(MODELS):
                e = load(run, f"eval_{split}.json")
                vals = [e["aligned"][d][metric] for d in doms] if e else [np.nan] * 3
                bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w * 0.92, color=color, label=label)
                bar_labels(ax, bars, "{:.2f}" if metric == "delta1" else "{:.3f}", 6.3)
            ax.set_xticks(range(3))
            ax.set_xticklabels(doms)
            ax.set_ylabel(nice)
            ax.set_title(f"{split} split — {nice}")
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.28))
    fig.text(0.5, -0.01, "All models scored after per-image least-squares scale/shift alignment in disparity "
             "space (seed 42). val = scenes 00001 + 00017, test = scenes 00000 + 00018.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "a2_results_val_test.png")


# ---------------------------------------------------------------- controlled experiment (seeds)
def seed_values(run, split, dom="overall"):
    out = {}
    for s in (42, 43, 44):
        e = load(run if s == 42 else f"{run}_seed{s}", f"eval_{split}.json")
        if e:
            out[s] = e["aligned"][dom]["abs_rel"]
    return out


def fig_seeds():
    runs = [m for m in MODELS if m[1] in ("midas_full", "midas_frozen")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, dom in zip(axes, ("overall", "indoor", "outdoor")):
        for k, split in enumerate(("val", "test")):
            for j, (label, run, color) in enumerate(runs):
                v = seed_values(run, split, dom)
                x = k * 3 + j
                ys = np.array(list(v.values()))
                ax.scatter([x] * len(ys), ys, s=28, color=color, edgecolor=SURFACE, lw=1.5, zorder=3,
                           label=label if (k == 0 and dom == "overall") else None)
                if len(ys) > 1:
                    m, sd = ys.mean(), ys.std(ddof=1)
                    ax.errorbar(x + 0.25, m, yerr=sd, fmt="_", color=INK, ms=12, capsize=3, lw=1.2)
                    ax.text(x + 0.38, m, f"{m:.4f}\n±{sd:.4f}", fontsize=6.5, va="center", color=INK)
        ax.set_xticks([0.5, 3.5])
        ax.set_xticklabels([f"val", f"test"])
        ax.set_xlim(-0.6, 5.2)
        ax.set_title(f"{dom} Abs Rel ↓")
    n = max(len(seed_values("midas_full", "val")), len(seed_values("midas_frozen", "val")))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Controlled experiment: full fine-tune vs frozen encoder — dots = seeds ({n} per config), "
                 "bar = mean ± std", fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "a2_controlled_seeds.png")


# ---------------------------------------------------------------- compute cost
def fig_compute():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    for label, run, color in MODELS:
        e = load(run, "eval_test.json")
        ax.scatter(e["infer_ms_per_image"], e["aligned"]["overall"]["abs_rel"], s=40 + 4 * e["params_M"],
                   color=color, edgecolor=SURFACE, lw=2, zorder=3)
        off = {"midas_full": (-70, 10), "midas_frozen": (10, 10)}.get(run, (8, -4))
        ax.annotate(f"{label}\n{e['params_M']:.1f}M params", (e["infer_ms_per_image"], e["aligned"]["overall"]["abs_rel"]),
                    xytext=off, textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("inference ms / image (batch 1, RTX 4060 Laptop, AMP)")
    ax.set_ylabel("test Abs Rel (overall) ↓")
    ax.set_xlim(0, 15)
    ax.set_ylim(0.22, 0.32)
    ax.grid(True, axis="both")
    ax.set_title("Accuracy vs latency (marker area ∝ params)")
    trained = [m for m in MODELS if m[1] != "midas_zeroshot"]
    for ax, key, nice, fmt in ((axes[1], "train_time_min", "training time (min)", "{:.1f}"),
                               (axes[2], "peak_vram_gb", "peak training VRAM (GB)", "{:.2f}")):
        for i, (label, run, color) in enumerate(trained):
            s = load(run, "train_summary.json")
            bars = ax.bar(i, s[key], 0.6, color=color)
            bar_labels(ax, bars, fmt, 8)
        ax.set_xticks(range(len(trained)))
        ax.set_xticklabels([m[0] for m in trained], fontsize=8)
        ax.set_title(nice)
    axes[1].text(0, -0.22, "U-Net: 40 epochs; MiDaS: 10 epochs", transform=axes[1].transAxes, fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "a2_compute_cost.png")


# ---------------------------------------------------------------- error analysis
def fig_error_distance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    bins = ["near (<5m)", "mid (5-20m)", "far (>20m)"]
    w = 0.2
    for ax, dom in zip(axes, ("indoor", "outdoor")):
        share = None
        for i, (label, run, color) in enumerate(MODELS):
            e = load(run, "eval_test.json")
            bb = e["aligned"]["bins"][dom]
            vals = [bb[b]["abs_rel"] if bb.get(b) else np.nan for b in bins]
            tot = sum(bb[b]["n_pixels"] for b in bins if bb.get(b))
            share = [100 * bb[b]["n_pixels"] / tot if bb.get(b) else 0 for b in bins]
            bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w * 0.92, color=color, label=label)
            bar_labels(ax, bars, "{:.2f}", 6.3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{b}\n{s:.1f}% of px" for b, s in zip(bins, share)])
        ax.set_title(f"test {dom}: Abs Rel by GT distance (pixel-pooled)")
    axes[0].set_ylabel("Abs Rel ↓")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "a2_error_by_distance_test.png")


def fig_error_analysis():
    with open(os.path.join(OUT, "diagnostics", "per_image_test.csv")) as f:
        rows = list(csv.DictReader(f))
    edge = {s: json.load(open(os.path.join(OUT, "diagnostics", f"edge_alignment_{s}.json")))
            for s in ("val", "test") if os.path.exists(os.path.join(OUT, "diagnostics", f"edge_alignment_{s}.json"))}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))

    # (a) outdoor error vs valid fraction
    ax = axes[0]
    out = [r for r in rows if r["domain"] == "outdoor"]
    vf = np.array([float(r["valid_frac"]) for r in out])
    for label, run, color in MODELS:
        if run not in ("midas_zeroshot", "midas_full"):
            continue
        a = np.array([float(r[f"{run}_abs_rel"]) for r in out])
        ax.scatter(100 * vf, a, s=9, color=color, alpha=0.45, lw=0)
        edges = np.array([0, 0.5, 0.75, 1.01])
        mids, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (vf >= lo) & (vf < hi)
            mids.append(100 * vf[s].mean())
            means.append(a[s].mean())
        ax.plot(mids, means, color=color, marker="o", ms=6, label=f"{label} (bin mean)")
    ax.set_xlabel("valid GT pixels in image (%)")
    ax.set_ylabel("per-image Abs Rel ↓")
    ax.set_title("(a) test outdoor: error vs. sky/invalid share")
    ax.grid(True, axis="both")
    ax.legend(fontsize=7.5)

    # (b) per-image distribution (ECDF) test, indoor and outdoor
    ax = axes[1]
    for label, run, color in MODELS:
        for dom, ls in (("indoor", "-"), ("outdoor", "--")):
            a = np.sort([float(r[f"{run}_abs_rel"]) for r in rows if r["domain"] == dom])
            ax.plot(a, np.arange(1, len(a) + 1) / len(a), color=color, ls=ls, lw=1.6,
                    label=label if dom == "indoor" else None)
    ax.axvline(0.25, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("per-image Abs Rel")
    ax.set_ylabel("fraction of test images ≤ x")
    ax.set_title("(b) per-image error CDF (solid indoor, dashed outdoor)")
    ax.grid(True, axis="both")
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) edge vs flat (test)
    ax = axes[2]
    e = edge.get("test")
    w = 0.2
    cats = [("indoor", "flat_abs_rel"), ("indoor", "edge_abs_rel"), ("outdoor", "flat_abs_rel"), ("outdoor", "edge_abs_rel")]
    for i, (label, run, color) in enumerate(MODELS):
        vals = [e[run][d][k] if e and run in e else np.nan for d, k in cats]
        bars = ax.bar(np.arange(4) + (i - 1.5) * w, vals, w * 0.92, color=color, label=label)
        bar_labels(ax, bars, "{:.2f}", 6)
    pct = {d: 100 * e["midas_full"][d]["edge_frac"] for d in ("indoor", "outdoor")} if e else {}
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"indoor\nflat", f"indoor\nedge ({pct.get('indoor', 0):.0f}% px)",
                        f"outdoor\nflat", f"outdoor\nedge ({pct.get('outdoor', 0):.0f}% px)"], fontsize=8)
    ax.set_ylabel("Abs Rel ↓")
    ax.set_title("(c) test: depth-edge vs. flat pixels")
    fig.tight_layout()
    save(fig, "a2_error_analysis.png")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    which = sys.argv[1:] or ["pipeline", "batch", "curves", "results", "seeds", "compute", "distance", "errors"]
    funcs = {"pipeline": fig_pipeline, "batch": fig_example_batch, "curves": fig_curves, "results": fig_results,
             "seeds": fig_seeds, "compute": fig_compute, "distance": fig_error_distance, "errors": fig_error_analysis}
    for k in which:
        funcs[k]()


if __name__ == "__main__":
    main()
