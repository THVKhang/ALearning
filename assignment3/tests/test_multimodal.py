"""Unit tests for Assignment 3 multimodal encoders, fusion modules, loss, and metrics."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.engine.loss import MultimodalJointLoss
from src.models.multimodal import build_model
from src.utils.metrics import compute_classification_metrics


def _dummy_batch(b=4, h=72, w=96):
    return {
        "rgb": torch.randn(b, 3, h, w),
        "depth_geo": torch.randn(b, 3, h, w),
        "valid_ratio": torch.rand(b, 1),
        "label": torch.tensor([0, 1, 4, 5][:b], dtype=torch.long),
    }


def test_four_mandatory_models_forward_backward():
    batch = _dummy_batch()
    criterion = MultimodalJointLoss(aux_weight=0.3)
    for mtype in ("rgb_only", "depth_only", "simple_fusion", "improved_fusion"):
        cfg = {
            "model": mtype,
            "use_cross_attn": True,
            "use_validity_gate": True,
            "train": {"modality_dropout": 0.15},
        }
        model = build_model(cfg)
        out = model(batch)
        assert out["logits"].shape == (4, 6), f"{mtype}: wrong logits shape {out['logits'].shape}"
        loss = criterion(out, batch["label"])
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, f"{mtype}: no gradients computed"
        print(f"ok {mtype:16s} params={sum(p.numel() for p in model.parameters()):,}")


def test_metrics_perfect_and_subtasks():
    y_true = [0, 1, 2, 3, 4, 5]
    m = compute_classification_metrics(y_true, y_true)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["domain_accuracy"] == 1.0 and m["elevation_accuracy"] == 1.0

    # Wrong elevation inside same domain -> domain_accuracy stays 1.0, elevation_accuracy drops
    y_same_dom = [1, 2, 0, 4, 5, 3]
    m2 = compute_classification_metrics(y_true, y_same_dom)
    assert m2["domain_accuracy"] == 1.0 and m2["elevation_accuracy"] == 0.0
    print("ok metrics")


if __name__ == "__main__":
    test_four_mandatory_models_forward_backward()
    test_metrics_perfect_and_subtasks()
    print("all assignment3 tests passed")
