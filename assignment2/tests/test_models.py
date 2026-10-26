"""Shape / gradient checks for the models. Run: python tests/test_models.py"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.unet import UNet


def test_unet():
    model = UNet()
    x = torch.randn(2, 3, 288, 384)
    out = model(x)
    assert out.shape == (2, 1, 288, 384)
    assert (out > 0).all()
    out.mean().backward()
    assert model.inc.conv[0].weight.grad is not None
    print(f"unet ok, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")


def test_midas_freeze():
    from src.models.midas import MiDaSSmall
    full, frozen = MiDaSSmall(False), MiDaSSmall(True)
    n_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    assert n_frozen < n_full
    assert not any(p.requires_grad for p in frozen.net.pretrained.parameters())
    assert all(p.requires_grad for p in frozen.net.scratch.parameters())
    frozen.train()
    assert not frozen.net.pretrained.training and frozen.net.scratch.training
    out = full.eval()(torch.randn(1, 3, 288, 384))
    assert out.shape == (1, 1, 288, 384)
    print(f"midas ok, trainable full {n_full / 1e6:.2f}M vs frozen {n_frozen / 1e6:.2f}M")


if __name__ == "__main__":
    test_unet()
    test_midas_freeze()