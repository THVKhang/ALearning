"""Compare least-squares vs. robust median/MAD disparity alignment and
decompose Abs Rel into depth-discontinuity edges vs. planar/smooth regions.

Usage:
    python scripts/report/edge_and_alignment_check.py --split val
    python scripts/report/edge_and_alignment_check.py --split test
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.data.dataset import MAX_DEPTH, DiodeDepthDataset
from src.engine.align import align_disparity
from src.engine.metrics import MIN_DEPTH, depth_metrics, valid_mask
from src.models import build_model
from src.utils.common import load_config, save_json


def align_disparity_median(pred_disp, gt, mask, max_depth):
    """Robust alignment in disparity space using median and mean absolute deviation (MAD)."""
    valid = valid_mask(gt, mask, max_depth)
    if not valid.any():
        return torch.full_like(gt, MIN_DEPTH)
    p = pred_disp[valid].float()
    y = 1.0 / gt[valid].float()
    med_p, med_y = p.median(), y.median()
    mad_p = (p - med_p).abs().mean().clamp(min=1e-6)
    mad_y = (y - med_y).abs().mean()
    s = mad_y / mad_p
    t = med_y - s * med_p
    disp = (s * pred_disp.float() + t).clamp(1.0 / max_depth, 1.0 / MIN_DEPTH)
    return 1.0 / disp


def depth_edge_mask(gt, valid, rel_thresh=0.10):
    """Split valid pixels into depth-discontinuity edges vs. smooth interior regions.
    A valid pixel is marked as an edge if any 4-connected valid neighbour differs in depth
    by more than `rel_thresh` (10% relative depth jump) or if it borders an invalid pixel.
    """
    g = gt.float()
    v = valid.bool()
    edge = torch.zeros_like(v)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        g_nb = torch.roll(g, shifts=(dy, dx), dims=(0, 1))
        v_nb = torch.roll(v, shifts=(dy, dx), dims=(0, 1))
        jump = (g - g_nb).abs() / g.clamp(min=MIN_DEPTH) > rel_thresh
        edge |= (~v_nb) | (v_nb & jump)
    edge_valid = v & edge
    flat_valid = v & (~edge)
    return edge_valid, flat_valid


def eval_subset(ds, model, output_type, device):
    stats = {
        dom: {
            "ls_abs_rel": [],
            "med_abs_rel": [],
            "edge_abs_rel": [],
            "flat_abs_rel": [],
            "edge_frac": [],
        }
        for dom in ("indoor", "outdoor")
    }
    for i in range(len(ds)):
        sample = ds[i]
        gt, mask = sample["depth"][0], sample["mask"][0]
        dom = sample["domain"]
        max_d = MAX_DEPTH[dom]
        v = valid_mask(gt, mask, max_d)
        if v.sum() < 100:
            continue

        with torch.no_grad():
            pred = model(sample["rgb"][None].to(device))[0, 0].float().cpu()

        disp_in = 1.0 / pred.clamp(min=MIN_DEPTH) if output_type == "metric" else pred
        pred_ls, _ = align_disparity(disp_in, gt, mask, max_d)
        pred_med = align_disparity_median(disp_in, gt, mask, max_d)

        m_ls = depth_metrics(pred_ls, gt, mask, max_d)
        m_med = depth_metrics(pred_med, gt, mask, max_d)
        if m_ls and m_med:
            stats[dom]["ls_abs_rel"].append(m_ls["abs_rel"])
            stats[dom]["med_abs_rel"].append(m_med["abs_rel"])

        edge_v, flat_v = depth_edge_mask(gt, v)
        m_edge = depth_metrics(pred_ls, gt, edge_v.float(), max_d)
        m_flat = depth_metrics(pred_ls, gt, flat_v.float(), max_d)
        if m_edge and m_flat:
            stats[dom]["edge_abs_rel"].append(m_edge["abs_rel"])
            stats[dom]["flat_abs_rel"].append(m_flat["abs_rel"])
            stats[dom]["edge_frac"].append(float(edge_v.sum()) / float(v.sum()))

    out = {}
    for dom, d in stats.items():
        out[dom] = {k: (sum(vals) / len(vals) if vals else float("nan")) for k, vals in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DiodeDepthDataset(os.path.join(ROOT_DIR, "data/diode_subset"), args.split)

    configs = [
        ("baseline_unet", "configs/baseline_unet.yaml"),
        ("midas_zeroshot", "configs/midas_zeroshot.yaml"),
        ("midas_full", "configs/midas_full.yaml"),
        ("midas_frozen", "configs/midas_frozen.yaml"),
    ]

    print(f"Split: {args.split} ({len(ds)} samples) on {device}")
    print(f"{'Model':<16} | {'Domain':<8} | {'LS Align':<9} | {'Med Align':<9} | {'Flat Reg':<9} | {'Edge Reg':<9} | {'Edge %':<7}")
    print("-" * 82)

    all_res = {}
    for name, cfg_rel in configs:
        cfg = load_config(os.path.join(ROOT_DIR, cfg_rel))
        ckpt = os.path.join(ROOT_DIR, cfg["out_dir"], "best.pth")
        if name != "midas_zeroshot" and not os.path.exists(ckpt):
            continue
        model, output_type = build_model(cfg)
        if name != "midas_zeroshot":
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
        model = model.to(device).eval()

        res = eval_subset(ds, model, output_type, device)
        all_res[name] = res
        for dom in ("indoor", "outdoor"):
            r = res[dom]
            print(f"{name:<16} | {dom:<8} | {r['ls_abs_rel']:<9.3f} | {r['med_abs_rel']:<9.3f} | "
                  f"{r['flat_abs_rel']:<9.3f} | {r['edge_abs_rel']:<9.3f} | {100*r['edge_frac']:<6.1f}%")

    save_json(all_res, os.path.join(ROOT_DIR, "outputs", "diagnostics", f"edge_alignment_{args.split}.json"))


if __name__ == "__main__":
    main()
