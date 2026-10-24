
"""Unit tests for loss functions and evaluation metrics."""

import math
import os
import sys
import numpy as np
import torch

# Ensure that src/engine can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.loss import MaskedSILogLoss
from src.engine.metrics import DepthMetricTracker, compute_depth_metrics


def test_silog_loss():
    print("[1/4] Testing MaskedSILogLoss...")
    criterion = MaskedSILogLoss(lambda_factor=0.5)

    # 1. Perfect prediction -> Loss = 0
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    loss = criterion(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), (
        f"Error: Expected 0, got {loss.item()}"
    )

    # 2. Check the backward pass and gradient computation
    pred = torch.tensor([[[[1.5, 2.0], [3.0, 4.0]]]], requires_grad=True)
    loss = criterion(pred, target, mask)
    loss.backward()
    assert pred.grad is not None, "Error: Gradient was not computed!"
    assert not torch.isnan(pred.grad).any(), "Error: Gradient contains NaN values!"

    # 3. Check mask filtering: the invalid pixel at (0, 1)
    # must not affect the loss
    criterion_no_reg = MaskedSILogLoss(lambda_factor=0.0)
    pred1 = torch.tensor([[[[2.0, 2.0], [3.0, 4.0]]]])
    pred2 = torch.tensor([[[[2.0, 999.0], [3.0, 4.0]]]])  # Extremely large error at (0, 1)
    mask_ignore = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # Ignore pixel (0, 1)

    l1 = criterion_no_reg(pred1, target, mask_ignore)
    l2 = criterion_no_reg(pred2, target, mask_ignore)
    assert torch.isclose(l1, l2, atol=1e-5), (
        f"Error: The mask failed to filter out the invalid pixel! {l1} vs {l2}"
    )

    # 4. Check scale invariance when lambda = 1.0
    scale_inv_criterion = MaskedSILogLoss(lambda_factor=1.0)
    pred_scaled = target * 3.5
    loss_scaled = scale_inv_criterion(pred_scaled, target, mask)
    assert torch.isclose(loss_scaled, torch.tensor(0.0), atol=1e-5), (
        "Error: SILog with lambda=1.0 should be zero when predictions "
        "differ only by a scale factor!"
    )

    print("  -> MaskedSILogLoss PASSED ALL TESTS!")


def test_metrics_exact_values():
    print("[2/4] Testing metric calculations against expected values...")

    # Target: [1.0, 2.0, 4.0]
    # Pred:   [1.2, 2.4, 4.8] (Exactly 1.2 times the target)
    # Ratio = 1.2 < 1.25 -> delta_1, delta_2, and delta_3 must all equal 1.0 (100%)
    # Abs Rel = |1.2 - 1.0| / 1.0 = 0.2 (20%)
    target = np.array([[[1.0, 2.0, 4.0]]], dtype=np.float32)
    pred = target * 1.2
    mask = np.ones_like(target)

    m = compute_depth_metrics(pred, target, mask)
    assert math.isclose(m["abs_rel"], 0.2, abs_tol=1e-4), (
        f"Error in Abs Rel: {m['abs_rel']}"
    )
    assert math.isclose(m["delta_1"], 1.0, abs_tol=1e-4), (
        f"Error in Delta 1: {m['delta_1']}"
    )
    assert math.isclose(m["delta_2"], 1.0, abs_tol=1e-4), (
        f"Error in Delta 2: {m['delta_2']}"
    )
    assert math.isclose(m["delta_3"], 1.0, abs_tol=1e-4), (
        f"Error in Delta 3: {m['delta_3']}"
    )

    # Check the case where the ratio = 1.4:
    # 1.4 > 1.25 -> delta_1 must equal 0.0
    # 1.4 < 1.5625 -> delta_2 must equal 1.0
    pred_14 = target * 1.4
    m2 = compute_depth_metrics(pred_14, target, mask)
    assert math.isclose(m2["delta_1"], 0.0, abs_tol=1e-4), (
        "Error: delta_1 should equal 0!"
    )
    assert math.isclose(m2["delta_2"], 1.0, abs_tol=1e-4), (
        "Error: delta_2 should equal 1.0!"
    )

    print("  -> All metric calculations are EXACTLY CORRECT!")


def test_metric_tracker():
    print("[3/4] Testing DepthMetricTracker...")
    tracker = DepthMetricTracker()

    b_pred = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]]],  # Sample 0: indoor, perfect prediction (abs_rel = 0.0)
        [[[2.0, 4.0], [6.0, 8.0]]],  # Sample 1: outdoor, predictions are twice the target (abs_rel = 1.0)
    ], dtype=torch.float32)

    b_target = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]]],
        [[[1.0, 2.0], [3.0, 4.0]]],
    ], dtype=torch.float32)

    b_mask = torch.ones_like(b_target)
    domains = ["indoor", "outdoor"]

    tracker.update(b_pred, b_target, b_mask, domains)
    summary = tracker.summary()

    assert summary["indoor"]["count"] == 1
    assert summary["outdoor"]["count"] == 1
    assert summary["overall"]["count"] == 2
    assert math.isclose(summary["indoor"]["abs_rel"], 0.0, abs_tol=1e-4)
    assert math.isclose(summary["outdoor"]["abs_rel"], 1.0, abs_tol=1e-4)
    assert math.isclose(summary["overall"]["abs_rel"], 0.5, abs_tol=1e-4)

    print("  -> DepthMetricTracker PASSED!")


def test_edge_cases():
    print("[4/4] Testing edge cases (NaNs, Infs, and empty masks)...")
    pred = np.array([[[float("nan"), 2.0], [0.0, float("inf")]]])
    target = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[[1, 1], [1, 1]]])

    # Only pixel (0, 1) is valid
    m = compute_depth_metrics(pred, target, mask)
    assert m["n_valid_pixels"] == 1, (
        f"Expected 1 valid pixel, got {m['n_valid_pixels']}"
    )

    # Mask containing only zeros
    mask_none = np.zeros_like(mask)
    m_none = compute_depth_metrics(pred, target, mask_none)
    assert m_none["n_valid_pixels"] == 0
    assert math.isnan(m_none["abs_rel"])

    print("  -> Edge cases handled SAFELY!")


if __name__ == "__main__":
    test_silog_loss()
    test_metrics_exact_values()
    test_metric_tracker()
    test_edge_cases()
    print("\n>>> CONGRATULATIONS: ALL TESTS PASSED SUCCESSFULLY! <<<")
