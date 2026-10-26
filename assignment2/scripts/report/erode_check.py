"""Re-evaluate validation predictions under eroded valid masks (0, 1, 2 px).

Usage:
    python scripts/report/erode_check.py
"""

import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.data.dataset import DiodeDepthDataset
from src.engine.align import align_disparity
from src.engine.metrics import depth_metrics, valid_mask
from src.models import build_model
from src.utils.common import load_config


def erode_mask(valid, k):
    """Erode valid mask by k pixels using 8-neighbourhood."""
    if k == 0:
        return valid
    inv = (~valid).float()[None, None]
    inv_dilated = F.max_pool2d(inv, kernel_size=2 * k + 1, stride=1, padding=k)
    return valid & (inv_dilated[0, 0] == 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DiodeDepthDataset(os.path.join(ROOT_DIR, "data/diode_subset"), "val")

    models = [
        ("baseline_unet", "configs/baseline_unet.yaml"),
        ("midas_zeroshot", "configs/midas_zeroshot.yaml"),
        ("midas_full", "configs/midas_full.yaml"),
        ("midas_frozen", "configs/midas_frozen.yaml"),
    ]

    print(f"Loaded {len(ds)} val samples.")
    print(f"{'Model':<16} | {'Domain':<8} | {'Orig (0px)':<10} | {'Erode 1px':<10} | {'Erode 2px':<10}")
    print("-" * 65)

    for name, cfg_path in models:
        cfg = load_config(os.path.join(ROOT_DIR, cfg_path))
        model, output_type = build_model(cfg)
        ckpt_path = os.path.join(ROOT_DIR, cfg["out_dir"], "best.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])

        model = model.to(device).eval()
        scores = {k: {"indoor": [], "outdoor": []} for k in (0, 1, 2)}

        for i in range(len(ds)):
            sample = ds[i]
            gt, mask = sample["depth"][0], sample["mask"][0]
            domain = "indoor" if "_indoors_" in ds.ids[i] else "outdoor"
            max_depth = 50.0 if domain == "indoor" else 200.0

            with torch.no_grad():
                pred = model(sample["rgb"][None].to(device))[0, 0].float().cpu()

            disp_input = 1.0 / pred.clamp(min=1e-3) if output_type == "metric" else pred
            pred_aligned, _ = align_disparity(disp_input, gt, mask, max_depth)

            v = valid_mask(gt, mask, max_depth)
            for k in (0, 1, 2):
                m_eroded = erode_mask(v, k).float()
                res = depth_metrics(pred_aligned, gt, m_eroded, max_depth)
                if res is not None:
                    scores[k][domain].append(res["abs_rel"])

        for dom in ["outdoor", "indoor"]:
            s0 = sum(scores[0][dom]) / len(scores[0][dom])
            s1 = sum(scores[1][dom]) / len(scores[1][dom])
            s2 = sum(scores[2][dom]) / len(scores[2][dom])
            print(f"{name:<16} | {dom:<8} | {s0:<10.3f} | {s1:<10.3f} | {s2:<10.3f}")


if __name__ == "__main__":
    main()
