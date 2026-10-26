"""Collect eval/train json of all runs into tables + figures in results/.

python scripts/report/make_report.py
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs")
RES = os.path.join(ROOT, "results")

# (label, run folder, which metrics: raw metres or aligned)
ROWS = [
    ("U-Net baseline (metric)", "baseline_unet", "raw"),
    ("U-Net baseline (aligned)", "baseline_unet", "aligned"),
    ("MiDaS_small zero-shot (aligned)", "midas_zeroshot", "aligned"),
    ("MiDaS_small full fine-tune (aligned)", "midas_full", "aligned"),
    ("MiDaS_small frozen encoder (aligned)", "midas_frozen", "aligned"),
]
TRAINED = [("U-Net baseline", "baseline_unet"), ("MiDaS full FT", "midas_full"),
           ("MiDaS frozen enc", "midas_frozen")]
DOMS = ("overall", "indoor", "outdoor")


def load(run, name):
    path = os.path.join(OUT, run, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def html_table(header, rows):
    h = "".join(f"<th>{c}</th>" for c in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>\n<tr>{h}</tr>\n{body}\n</table>"


def results_table(split):
    header = ["Model"] + [f"{d} {m}" for d in DOMS for m in ("AbsRel↓", "RMSE↓", "δ1↑")]
    rows = []
    for label, run, kind in ROWS:
        ev = load(run, f"eval_{split}.json")
        if ev is None or kind not in ev:
            rows.append([label] + ["in progress"] * 9)
            continue
        r = [label]
        for d in DOMS:
            m = ev[kind][d]
            r += [f"{m['abs_rel']:.3f}", f"{m['rmse']:.2f}", f"{100 * m['delta1']:.1f}%"]
        rows.append(r)
    return header, rows


def cost_table():
    header = ["Model", "Params (M)", "Trainable (M)", "Epochs", "Train time (min)",
              "s / epoch", "Peak train VRAM (GB)", "Inference (ms/img, bs1)"]
    rows, env = [], None
    runs = [("U-Net baseline", "baseline_unet"), ("MiDaS_small zero-shot", "midas_zeroshot"),
            ("MiDaS_small full fine-tune", "midas_full"), ("MiDaS_small frozen encoder", "midas_frozen")]
    for name, run in runs:
        ev, tr = load(run, "eval_val.json"), load(run, "train_summary.json")
        if ev is None:
            continue
        env = env or ev["env"]
        if tr:
            rows.append([name, f"{tr['params_M']:.2f}", f"{tr['trainable_params_M']:.2f}", tr["epochs_run"],
                         f"{tr['train_time_min']:.1f}", f"{tr['epoch_time_s']:.0f}",
                         f"{tr['peak_vram_gb']:.2f}", f"{ev['infer_ms_per_image']:.1f}"])
        else:
            rows.append([name, f"{ev['params_M']:.2f}", "0", "-", "-", "-", "-",
                         f"{ev['infer_ms_per_image']:.1f}"])
    return header, rows, env


def error_by_distance(split):
    header = ["Model", "Domain", "Bin", "AbsRel", "RMSE", "δ1", "% of pixels"]
    rows, plot = [], {}
    for label, run, kind in ROWS:
        if kind == "raw":
            continue
        ev = load(run, f"eval_{split}.json")
        if ev is None:
            continue
        name = label.replace(" (aligned)", "")
        for dom, bins in ev[kind]["bins"].items():
            total = sum(b["n_pixels"] for b in bins.values() if b)
            for b, m in bins.items():
                if m is None:
                    continue
                rows.append([name, dom, b, f"{m['abs_rel']:.3f}", f"{m['rmse']:.2f}",
                             f"{100 * m['delta1']:.1f}%", f"{100 * m['n_pixels'] / total:.1f}%"])
                plot.setdefault((dom, b), {})[name] = m["abs_rel"]
    return header, rows, plot


def plot_errors(plot, path):
    if not plot:
        return
    keys = list(plot)
    models = list(dict.fromkeys(m for v in plot.values() for m in v))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(models):
        ax.bar([k + i * w for k in range(len(keys))], [plot[k].get(m, 0) for k in keys], w, label=m)
    ax.set_xticks([k + w * (len(models) - 1) / 2 for k in range(len(keys))])
    ax.set_xticklabels([f"{d}\n{b}" for d, b in keys], fontsize=8)
    ax.set_ylabel("Abs Rel (aligned)")
    ax.set_title("Val error by GT distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (label, run) in zip(axes, TRAINED):
        hp = os.path.join(OUT, run, "history.csv")
        if not os.path.exists(hp):
            ax.set_title(f"{label} (in progress)")
            continue
        with open(hp) as f:
            h = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in h]
        ax.plot(ep, [float(r["train_loss"]) for r in h], label="train loss")
        ax.plot(ep, [float(r["val_loss"]) for r in h], label="val loss")
        ax2 = ax.twinx()
        ax2.plot(ep, [float(r["overall_abs_rel"]) for r in h], "g--", label="val AbsRel")
        ax2.set_ylabel("val Abs Rel", color="g")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    md, html = [], []

    h, r = results_table("val")
    md += ["## Results (val)", md_table(h, r)]
    html += [html_table(h, r)]

    h, r, env = cost_table()
    md += ["## Compute cost", md_table(h, r)]
    html += [html_table(h, r)]
    if env:
        line = (f"GPU: {env['gpu']} | Python {env['python']} | PyTorch {env['torch']} | CUDA {env['cuda']} "
                f"| cuDNN {env['cudnn']} | timm {env.get('timm', '-')}")
        md.append(line)
        html.append(f"<p>{line}</p>")

    h, r, plot = error_by_distance("val")
    md += ["## Error by distance (val, aligned)", md_table(h, r)]
    html += [html_table(h, r)]
    plot_errors(plot, os.path.join(RES, "error_by_distance.png"))
    plot_curves(os.path.join(RES, "loss_curves.png"))

    with open(os.path.join(RES, "tables.md"), "w") as f:
        f.write("\n\n".join(md) + "\n")
    with open(os.path.join(RES, "tables.html"), "w") as f:
        f.write("\n\n".join(html) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
