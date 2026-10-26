"""Zero-shot MiDaS on 25 outdoor images per scene: is the poor val-outdoor score the scene or the pipeline?
Prints rank/Pearson correlation of pred vs 1/GT and Abs Rel for two alignments, plus Abs Rel without boundary pixels.

python scripts/report/outdoor_check.py
"""

import sys
import numpy as np
import torch
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data.dataset import DiodeDepthDataset, MAX_DEPTH
from src.engine.align import align_disparity
from src.engine.metrics import depth_metrics, valid_mask
from src.models.midas import MiDaSSmall

ROOT = os.path.join(ROOT_DIR, "data/diode_subset")
m = MiDaSSmall().cuda().eval()


def ranks(x):
    return torch.argsort(torch.argsort(x)).float()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


print("scene split n | spearman(pred,1/gt) pearson | AbsRel: LS-disp  logfit  | edge-free LS-disp")
for split in ["train", "val", "test"]:
    ds = DiodeDepthDataset(ROOT, split)
    scenes = {}
    for i, s in enumerate(ds.ids):
        if "_indoors_" not in s:
            scenes.setdefault(s.split("_")[0], []).append(i)
    for sc, idx in scenes.items():
        idx = idx[:: max(1, len(idx) // 25)][:25]
        rows = []
        for i in idx:
            s = ds[i]
            gt, mask = s["depth"][0], s["mask"][0]
            with torch.no_grad():
                p = m(s["rgb"][None].cuda())[0, 0].float().cpu()
            v = valid_mask(gt, mask, 200.0)
            if v.sum() < 500:
                continue
            sp = corr(ranks(p[v]), ranks(1 / gt[v])).item()
            pe = corr(p[v], 1 / gt[v]).item()
            d, _ = align_disparity(p, gt, mask, 200.0)
            a1 = depth_metrics(d, gt, mask, 200.0)["abs_rel"]
            # power-law fit in log space: log gt = a*log(p) + b
            lp, lg = torch.log(p[v].clamp(min=1e-3)), torch.log(gt[v])
            A = torch.stack([lp, torch.ones_like(lp)], 1).double()
            sol = torch.linalg.lstsq(A, lg.double()[:, None]).solution[:, 0]
            pred = torch.exp(sol[0] * torch.log(p.clamp(min=1e-3)) + sol[1]).float()
            a2 = depth_metrics(pred, gt, mask, 200.0)["abs_rel"]
            # drop pixels next to invalid ones (thin structures / edges)
            inv = ~v
            nb = torch.zeros_like(inv)
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb |= torch.roll(inv, (dy, dx), (0, 1))
            m_in = (v & ~nb).float()[None]
            a3 = depth_metrics(d, gt, m_in[0], 200.0)
            rows.append((sp, pe, a1, a2, a3["abs_rel"] if a3 else float("nan")))
        r = np.nanmean(np.array(rows), 0)
        print(f"{sc} {split:5s} {len(rows):2d} | {r[0]:.2f} {r[1]:.2f} | {r[2]:.3f} {r[3]:.3f} | {r[4]:.3f}")
