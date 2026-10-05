import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DEPTH_SUFFIX, MASK_SUFFIX, SPLITS, list_samples

BINS = np.arange(0, 400.05, 0.05)  # depth histogram bins (m)
THRESH = {"indoor": [20, 30, 50], "outdoor": [100, 150, 200, 300]}
UP_CODE = 40  # vertical angle code >= 40 -> camera looks up at ceiling / sky


def parse_id(sid):
    # e.g. 00005_00039_indoors_250_050 -> scene, scan, domain, horizontal angle, vertical angle
    scene, scan, dom, h, v = sid.split("_")
    return scene, scan, "indoor" if dom == "indoors" else "outdoor", int(h), int(v)


def pct_from_hist(hist, q):
    cdf = np.cumsum(hist) / hist.sum()
    return BINS[np.searchsorted(cdf, q / 100) + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    ap.add_argument("--out", default="outputs/eda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    hists = defaultdict(lambda: np.zeros(len(BINS) - 1))
    for split in SPLITS:
        d = os.path.join(args.root, split)
        for sid in list_samples(d):
            scene, scan, dom, h, v = parse_id(sid)
            depth = np.load(os.path.join(d, sid + DEPTH_SUFFIX)).astype(np.float32).squeeze()
            mask = np.load(os.path.join(d, sid + MASK_SUFFIX)).squeeze()
            valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
            dv = depth[valid]
            gray = np.asarray(Image.open(os.path.join(d, sid + ".jpg")).convert("L"), dtype=np.float32)
            hists[(split, dom)] += np.histogram(dv, BINS)[0]

            r = {"id": sid, "split": split, "domain": dom, "scene": scene, "scan": scan,
                 "h_angle": h, "v_angle": v, "invalid_pct": 100 * (1 - valid.mean()),
                 "median": np.median(dv) if dv.size else np.nan,
                 "p99": np.percentile(dv, 99) if dv.size else np.nan,
                 "max": dv.max() if dv.size else np.nan, "brightness": gray.mean()}
            for t in THRESH[dom]:
                r[f"pct_gt_{t}"] = 100 * (dv > t).mean() if dv.size else 0.0
            rows.append(r)
        print(f"{split}: done")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(args.out, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    # 1) depth distribution + invalid % per split and domain
    out("## Depth and invalid pixels per split / domain")
    out("| split | domain | samples | depth p5 | median | p95 | p99 | p99.9 | invalid % (mean) | invalid % (median) | brightness |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for split in SPLITS:
        for dom in ("indoor", "outdoor"):
            s = sel(split=split, domain=dom)
            hst = hists[(split, dom)]
            p = [pct_from_hist(hst, q) for q in (5, 50, 95, 99, 99.9)]
            inv = np.array([r["invalid_pct"] for r in s])
            br = np.mean([r["brightness"] for r in s])
            out(f"| {split} | {dom} | {len(s)} | " + " | ".join(f"{x:.1f}" for x in p) +
                f" | {inv.mean():.1f} | {np.median(inv):.1f} | {br:.0f} |")

    # 2) outliers
    out("\n## Outliers (train)")
    out("| domain | threshold | samples with any pixel above | % of all valid pixels above |")
    out("|---|---|---|---|")
    for dom in ("indoor", "outdoor"):
        s = sel(split="train", domain=dom)
        hst = hists[("train", dom)]
        for t in THRESH[dom]:
            n_any = sum(r["max"] > t for r in s)
            frac = 100 * hst[BINS[:-1] >= t].sum() / hst.sum()
            out(f"| {dom} | > {t} m | {n_any} / {len(s)} ({100 * n_any / len(s):.1f}%) | {frac:.4f}% |")

    # 3) vertical angle per scene
    codes = sorted({r["v_angle"] for r in rows})
    out("\n## Vertical angle code per scene (% of samples)")
    out("| scene | split | domain | samples | " + " | ".join(f"{c:03d}" for c in codes) +
        f" | looks up (>= {UP_CODE:03d}) |")
    out("|---|---|---|---|" + "---|" * (len(codes) + 1))
    scenes = sorted({(r["split"], r["scene"], r["domain"]) for r in rows},
                    key=lambda x: (SPLITS.index(x[0]), x[2], x[1]))
    for split, scene, dom in scenes:
        s = sel(scene=scene)
        share = [100 * sum(r["v_angle"] == c for r in s) / len(s) for c in codes]
        up = 100 * sum(r["v_angle"] >= UP_CODE for r in s) / len(s)
        out(f"| {scene} | {split} | {dom} | {len(s)} | " + " | ".join(f"{x:.0f}" for x in share) + f" | {up:.0f}% |")
    for split in SPLITS:
        for dom in ("indoor", "outdoor"):
            s = sel(split=split, domain=dom)
            up = 100 * sum(r["v_angle"] >= UP_CODE for r in s) / len(s)
            out(f"- {split} {dom}: {up:.1f}% of samples look up")

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figures
    centers = (BINS[:-1] + BINS[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dom in zip(axes, ("indoor", "outdoor")):
        for split in SPLITS:
            hst = hists[(split, dom)]
            # merge into wider bins so the lines are not too noisy
            k = 10 if dom == "indoor" else 40
            n = len(hst) // k * k
            ax.plot(centers[:n].reshape(-1, k).mean(1), hst[:n].reshape(-1, k).sum(1) / hst.sum(), label=split)
        ax.set_xlim(0, 40 if dom == "indoor" else 250)
        ax.set_title(f"{dom} depth (valid pixels)")
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("fraction of pixels")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "depth_hist.png"), dpi=110)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dom in zip(axes, ("indoor", "outdoor")):
        for split in SPLITS:
            inv = [r["invalid_pct"] for r in sel(split=split, domain=dom)]
            ax.hist(inv, bins=np.arange(0, 101, 2), histtype="step", density=True, label=split, lw=1.5)
        ax.set_title(f"{dom}: invalid pixels per image")
        ax.set_xlabel("invalid %")
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "invalid_hist.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(12, 4))
    names = [f"{sc}\n{sp}" for sp, sc, _ in scenes]
    bottom = np.zeros(len(scenes))
    for c in codes:
        share = np.array([100 * sum(r["v_angle"] == c for r in sel(scene=sc)) / len(sel(scene=sc))
                          for _, sc, _ in scenes])
        ax.bar(names, share, bottom=bottom, label=f"{c:03d}")
        bottom += share
    ax.set_ylabel("% of samples")
    ax.set_title("vertical angle code per scene (000 = level/down, 050 = up)")
    ax.legend(title="code", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "angle_by_scene.png"), dpi=110)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
