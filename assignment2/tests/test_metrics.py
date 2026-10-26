"""Sanity checks for metrics, alignment and losses. Run: python tests/test_metrics.py (or pytest)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.engine.align import align_batch, align_disparity
from src.engine.loss import MaskedSILogLoss, ScaleShiftInvariantLoss
from src.engine.metrics import MetricTracker, depth_metrics


def fake_depth(b=2, h=24, w=32, lo=0.5, hi=40.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = lo + (hi - lo) * torch.rand(b, 1, h, w, generator=g)
    mask = (torch.rand(b, 1, h, w, generator=g) > 0.2).float()
    return depth * mask, mask


def test_gt_as_pred_is_perfect():
    gt, mask = fake_depth()
    m = depth_metrics(gt[0], gt[0], mask[0], max_depth=50.0)
    assert abs(m["abs_rel"]) < 1e-7 and abs(m["rmse"]) < 1e-6 and m["delta1"] == 1.0


def test_known_values():
    gt = torch.tensor([[[1.0, 2.0, 4.0]]])
    mask = torch.ones_like(gt)
    m = depth_metrics(gt * 1.2, gt, mask, 50.0)
    assert abs(m["abs_rel"] - 0.2) < 1e-6 and m["delta1"] == 1.0
    m = depth_metrics(gt * 1.3, gt, mask, 50.0)
    assert m["delta1"] == 0.0
    rmse = (((gt * 0.3) ** 2).mean()).sqrt().item()
    assert abs(m["rmse"] - rmse) < 1e-6


def test_cap_and_mask():
    gt = torch.tensor([[[2.0, 60.0, 3.0]]])     # 60 m is beyond the indoor cap
    mask = torch.tensor([[[1.0, 1.0, 0.0]]])    # last pixel invalid
    pred = torch.tensor([[[2.0, 999.0, 999.0]]])
    m = depth_metrics(pred, gt, mask, max_depth=50.0)
    assert m["abs_rel"] == 0.0


def test_bad_pred_is_penalised_not_dropped():
    gt = torch.tensor([[[2.0, 2.0]]])
    mask = torch.ones_like(gt)
    m = depth_metrics(torch.tensor([[[2.0, -5.0]]]), gt, mask, 50.0)
    assert m["delta1"] == 0.5 and m["abs_rel"] > 0.4


def test_tracker_domains():
    gt, mask = fake_depth(b=2)
    pred = gt.clone()
    pred[1] = gt[1] * 2
    t = MetricTracker()
    t.update(pred, gt, mask, ["indoor", "outdoor"])
    s = t.summary()
    assert s["indoor"]["abs_rel"] < 1e-7
    assert abs(s["outdoor"]["abs_rel"] - 1.0) < 1e-5
    assert abs(s["overall"]["abs_rel"] - 0.5) < 1e-5
    assert s["overall"]["n_images"] == 2


def test_alignment_recovers_gt():
    gt, mask = fake_depth(b=2, lo=1.0, hi=30.0)
    disp = torch.where(mask > 0, 1.0 / gt.clamp(min=1e-3), torch.zeros_like(gt))
    fake_midas = 123.0 * disp + 0.7           # invert, scale, shift
    out, (s, t) = align_disparity(fake_midas[0], gt[0], mask[0], max_depth=50.0)
    v = mask[0] > 0
    assert (out[v] - gt[0][v]).abs().max() < 1e-3, (out[v] - gt[0][v]).abs().max()
    m = depth_metrics(align_batch(fake_midas, gt, mask, ["indoor", "outdoor"])[1], gt[1], mask[1], 50.0)
    assert m["abs_rel"] < 1e-5 and m["delta1"] == 1.0


def test_silog():
    gt, mask = fake_depth()
    loss = MaskedSILogLoss(0.5)
    assert loss(gt.clamp(min=1e-3), gt, mask).abs() < 1e-6
    p = (gt * 1.5).clamp(min=1e-3).requires_grad_()
    loss(p, gt, mask).backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    p2 = gt.clone().clamp(min=1e-3)
    p2[mask == 0] = 1e4                       # invalid pixels must not matter
    assert loss(p2, gt, mask).abs() < 1e-6
    assert MaskedSILogLoss(1.0)((gt * 3).clamp(min=1e-3), gt, mask).abs() < 1e-5


def test_ssi_loss_is_scale_shift_invariant():
    gt, mask = fake_depth(lo=1.0, hi=30.0)
    disp = torch.where(mask > 0, 1.0 / gt.clamp(min=1e-3), torch.zeros_like(gt))
    loss = ScaleShiftInvariantLoss()
    assert loss(50.0 * disp + 3.0, gt, mask) < 1e-5
    assert loss(torch.rand_like(disp), gt, mask) > 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all passed")
