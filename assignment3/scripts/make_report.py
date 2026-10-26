"""Generate Assignment 3 comparison tables, support/conflict analysis, ablation tables,
learning curves, confusion matrices, and qualitative multimodal examples.
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.dataset import DiodeMultimodalDataset
from src.utils.common import save_json
from src.utils.metrics import CLASS_NAMES, SHORT_CLASS_NAMES

OUT = os.path.join(ROOT, "outputs")
ASSETS = os.path.join(ROOT, "assets")

MAIN_MODELS = [
    ("rgb_only", "Unimodal A: RGB-Only ConvNet"),
    ("depth_only", "Unimodal B: Depth-Geometry ConvNet"),
    ("simple_fusion", "Simple Fusion: Late Concatenation"),
    ("improved_fusion", "Improved Fusion: Cross-Attn + Validity Gate"),
]

ALL_MODELS = MAIN_MODELS + [
    ("ablation_no_gate", "Ablation 1: No Cross-Attn / No Validity Gate"),
    ("ablation_no_aux", "Ablation 2: No Aux Unimodal Loss / No Modality Drop"),
]


def load_json(run, fname):
    p = os.path.join(OUT, run, fname)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def analyze_support_and_conflict():
    """Compare test predictions of `rgb_only`, `depth_only`, `simple_fusion`, and `improved_fusion`."""
    p_rgb = load_json("rgb_only", "predictions_test.json")
    p_dep = load_json("depth_only", "predictions_test.json")
    p_sim = load_json("simple_fusion", "predictions_test.json")
    p_imp = load_json("improved_fusion", "predictions_test.json")

    y_true = np.asarray(p_rgb["y_true"])
    r_ok = np.asarray(p_rgb["y_pred"]) == y_true
    d_ok = np.asarray(p_dep["y_pred"]) == y_true
    s_ok = np.asarray(p_sim["y_pred"]) == y_true
    i_ok = np.asarray(p_imp["y_pred"]) == y_true

    regimes = [
        ("Both Unimodal Correct (RGB ✓, Depth ✓)", r_ok & d_ok),
        ("Conflict: RGB Correct, Depth Wrong (RGB ✓, Depth ✗)", r_ok & (~d_ok)),
        ("Conflict: Depth Correct, RGB Wrong (RGB ✗, Depth ✓)", (~r_ok) & d_ok),
        ("Both Unimodal Wrong — Emergent Support (RGB ✗, Depth ✗)", (~r_ok) & (~d_ok)),
    ]

    rows = []
    summary_dict = {}
    for label, mask in regimes:
        cnt = int(mask.sum())
        s_win = int((s_ok & mask).sum())
        i_win = int((i_ok & mask).sum())
        s_pct = 100.0 * s_win / max(cnt, 1)
        i_pct = 100.0 * i_win / max(cnt, 1)
        rows.append([label, f"{cnt} ({100*cnt/len(y_true):.1f}%)", f"{s_win}/{cnt} ({s_pct:.1f}%)", f"{i_win}/{cnt} ({i_pct:.1f}%)"])
        summary_dict[label] = {
            "count": cnt,
            "share_pct": round(100.0 * cnt / len(y_true), 2),
            "simple_fusion_correct": s_win,
            "simple_fusion_rate_pct": round(s_pct, 2),
            "improved_fusion_correct": i_win,
            "improved_fusion_rate_pct": round(i_pct, 2),
        }

    # Find concrete example indices for visualization
    idx_emergent = np.where((~r_ok) & (~d_ok) & i_ok)[0]
    idx_rgb_win = np.where(r_ok & (~d_ok) & i_ok)[0]
    idx_dep_win = np.where((~r_ok) & d_ok & i_ok)[0]
    idx_all_fail = np.where((~r_ok) & (~d_ok) & (~i_ok))[0]

    summary_dict["example_indices"] = {
        "emergent_support": int(idx_emergent[0]) if len(idx_emergent) else 0,
        "conflict_rgb_wins": int(idx_rgb_win[0]) if len(idx_rgb_win) else 1,
        "conflict_depth_wins": int(idx_dep_win[0]) if len(idx_dep_win) else 2,
        "shared_failure": int(idx_all_fail[0]) if len(idx_all_fail) else 3,
    }
    save_json(summary_dict, os.path.join(OUT, "support_conflict_test.json"))
    return rows, summary_dict, (p_rgb, p_dep, p_sim, p_imp)


def plot_learning_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for key, label in ALL_MODELS:
        hp = os.path.join(OUT, key, "history.csv")
        if not os.path.exists(hp):
            continue
        with open(hp, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        v_loss = [float(r["val_loss"]) for r in rows]
        v_f1 = [100.0 * float(r["val_macro_f1"]) for r in rows]
        ls = "--" if key.startswith("ablation") else "-"
        lw = 2.2 if key == "improved_fusion" else 1.5
        ax1.plot(ep, v_loss, linestyle=ls, linewidth=lw, label=label)
        ax2.plot(ep, v_f1, linestyle=ls, linewidth=lw, label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("A. Validation Loss Trajectory")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro-F1 (%)")
    ax2.set_title("B. Validation Macro-F1 Progression")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "curves_all_models.png"), dpi=140)
    plt.close(fig)


def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    for ax, (key, title) in zip(axes.flatten(), MAIN_MODELS):
        ev = load_json(key, "eval_test.json")
        cm = np.asarray(ev["confusion_matrix"], dtype=np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sums, 1.0)) * 100.0

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=100.0)
        ax.set_title(f"{title}\n(Test Acc: {100*ev['accuracy']:.1f}%, Macro-F1: {100*ev['macro_f1']:.1f}%)", fontsize=9.5)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT_CLASS_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(SHORT_CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted Class", fontsize=8.5)
        ax.set_ylabel("True Class", fontsize=8.5)

        for i in range(6):
            for j in range(6):
                val = cm_norm[i, j]
                cnt = int(cm[i, j])
                if cnt > 0:
                    color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.0f}%\n({cnt})", ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "confusion_matrices.png"), dpi=140)
    plt.close(fig)


def plot_qualitative_examples(sc_summary, preds_tuple):
    p_rgb, p_dep, p_sim, p_imp = preds_tuple
    data_root = os.path.abspath(os.path.join(ROOT, "..", "assignment2", "data", "diode_subset"))
    ds = DiodeMultimodalDataset(data_root, split="test", img_size=(72, 96), hflip=False)

    ex = sc_summary["example_indices"]
    cases = [
        ("1. Emergent Support\n(Both Unimodal ✗ → Fusion ✓)", ex["emergent_support"]),
        ("2. Conflict Resolved via RGB\n(RGB ✓, Depth ✗ → Fusion ✓)", ex["conflict_rgb_wins"]),
        ("3. Conflict Resolved via Depth\n(RGB ✗, Depth ✓ → Fusion ✓)", ex["conflict_depth_wins"]),
        ("4. Shared Failure Case\n(Ambiguous Boundary → All ✗)", ex["shared_failure"]),
    ]

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12.0, 9.8))
    for r, (case_title, idx) in enumerate(cases):
        s = ds[idx]
        rgb_vis = np.clip(s["rgb"].numpy() * std + mean, 0, 1).transpose(1, 2, 0)
        d_geo = s["depth_geo"].numpy()
        gt_lbl = SHORT_CLASS_NAMES[p_rgb["y_true"][idx]]
        rgb_lbl = SHORT_CLASS_NAMES[p_rgb["y_pred"][idx]]
        dep_lbl = SHORT_CLASS_NAMES[p_dep["y_pred"][idx]]
        sim_lbl = SHORT_CLASS_NAMES[p_sim["y_pred"][idx]]
        imp_lbl = SHORT_CLASS_NAMES[p_imp["y_pred"][idx]]
        v_rat = 100.0 * p_imp["valid_ratio"][idx]

        axes[r, 0].imshow(rgb_vis)
        axes[r, 0].set_ylabel(case_title, fontsize=8.5, fontweight="bold")
        axes[r, 1].imshow(d_geo[0], cmap="magma", vmin=0.0, vmax=0.85)
        axes[r, 2].imshow(d_geo[2], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[r, 3].axis("off")

        summary_txt = (
            f"Sample ID: {s['id']}\n"
            f"Valid LiDAR pixels: {v_rat:.1f}%\n"
            f"──────────────────────────\n"
            f"Ground Truth : {gt_lbl}\n"
            f"RGB-Only     : {rgb_lbl}\n"
            f"Depth-Only   : {dep_lbl}\n"
            f"Simple Concat: {sim_lbl}\n"
            f"Improved Fus.: {imp_lbl}"
        )
        axes[r, 3].text(
            0.03,
            0.5,
            summary_txt,
            va="center",
            ha="left",
            fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6fa", edgecolor="#cbd5e1"),
        )

        if r == 0:
            axes[r, 0].set_title("Modality A: RGB Image", fontsize=9.5)
            axes[r, 1].set_title("Modality B: Log-Depth Map", fontsize=9.5)
            axes[r, 2].set_title("Modality B: Vertical Slant ∂z/∂y", fontsize=9.5)
            axes[r, 3].set_title("Unimodal vs. Multimodal Predictions", fontsize=9.5)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "support_conflict_examples.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    res_dir = os.path.join(ROOT, "results")
    os.makedirs(res_dir, exist_ok=True)

    # 1. Main comparison table (Val & Test)
    main_headers = [
        "Model",
        "Params",
        "Val Acc",
        "Val Macro-F1",
        "Test Acc",
        "Test Macro-F1",
        "Test Domain Acc",
        "Test Elev Acc",
        "Train Time (s)",
        "Infer (ms/pair)",
    ]
    main_rows = []
    for key, label in ALL_MODELS:
        tr = load_json(key, "train_summary.json")
        ev_v = load_json(key, "eval_val.json")
        ev_t = load_json(key, "eval_test.json")
        if not (tr and ev_v and ev_t):
            continue
        main_rows.append(
            [
                label,
                f"{tr['params']:,}",
                f"{100*ev_v['accuracy']:.2f}%",
                f"{100*ev_v['macro_f1']:.2f}%",
                f"{100*ev_t['accuracy']:.2f}%",
                f"{100*ev_t['macro_f1']:.2f}%",
                f"{100*ev_t['domain_accuracy']:.2f}%",
                f"{100*ev_t['elevation_accuracy']:.2f}%",
                f"{tr['train_time_s']:.1f}s ({tr['epoch_time_s']:.2f}s/ep)",
                f"{ev_t['infer_ms_per_sample']:.3f}",
            ]
        )

    # 2. Per-class F1 on Test
    cls_headers = ["Class"] + [label.split(":")[0] for _, label in MAIN_MODELS]
    cls_rows = []
    for cname in CLASS_NAMES:
        scores = [load_json(k, "eval_test.json")["per_class_f1"][cname] for k, _ in MAIN_MODELS]
        best = max(scores)
        cells = [f"**{100*s:.2f}%**" if abs(s - best) < 1e-9 else f"{100*s:.2f}%" for s in scores]
        cls_rows.append([cname] + cells)

    # 3. Missing-Modality & Validity-Regime Robustness Table
    rob_headers = [
        "Fusion Model",
        "Full Test Macro-F1",
        "Clean LiDAR (≥65% valid)",
        "Noisy LiDAR (<65% valid)",
        "Drop Depth at Test (0% LiDAR)",
        "Drop RGB at Test (0% RGB)",
    ]
    rob_rows = []
    for key, label in MAIN_MODELS[2:] + ALL_MODELS[4:]:
        ev_t = load_json(key, "eval_test.json")
        vr = ev_t["by_validity_regime"]
        ms = ev_t["missing_modality_stress"]
        rob_rows.append(
            [
                label,
                f"{100*ev_t['macro_f1']:.2f}%",
                f"{100*vr['clean_lidar_ge_65pct_valid']:.2f}%",
                f"{100*vr['noisy_lidar_lt_65pct_valid']:.2f}%",
                f"{100*ms['drop_depth_at_test_macro_f1']:.2f}%",
                f"{100*ms['drop_rgb_at_test_macro_f1']:.2f}%",
            ]
        )

    # 4. Support vs. Conflict Table
    sc_rows, sc_summary, preds_tuple = analyze_support_and_conflict()
    sc_headers = [
        "Unimodal Agreement Regime (Test Split, n=1,100)",
        "Test Samples (%)",
        "Simple Fusion Correct",
        "Improved Fusion Correct",
    ]

    plot_learning_curves()
    plot_confusion_matrices()
    plot_qualitative_examples(sc_summary, preds_tuple)

    md_out = "\n\n".join(
        [
            "## 1. Main Quantitative Comparison (Val & Held-Out Test Splits)",
            md_table(main_headers, main_rows),
            "## 2. Per-Class F1 on Held-Out Test Split",
            md_table(cls_headers, cls_rows),
            "## 3. Cross-Modal Support vs. Conflict Breakdown on Held-Out Test Split",
            md_table(sc_headers, sc_rows),
            "## 4. Missing-Modality Stress Test & LiDAR Validity Regime Breakdown",
            md_table(rob_headers, rob_rows),
        ]
    )
    with open(os.path.join(res_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md_out + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md_out)


if __name__ == "__main__":
    main()
