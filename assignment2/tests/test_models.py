"""Smoke test for the baseline U-Net architecture."""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.unet import UNet


def test_unet():
    print("[1/2] Testing the U-Net architecture...")
    model = UNet(in_channels=3, out_channels=1)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Total trainable parameters: {n_params / 1e6:.2f}M parameters")

    # Create a dummy RGB batch matching the DIODE input size: 2 images, 3 channels, 288x384
    dummy_input = torch.randn(2, 3, 288, 384)
    print(f"  -> Input shape (RGB): {list(dummy_input.shape)}")

    # Forward pass
    output = model(dummy_input)
    print(f"  -> Output shape (Depth): {list(output.shape)}")

    assert output.shape == (2, 1, 288, 384), "Error: Output shape does not match the expected dimensions!"
    assert (output > 0).all(), "Error: Depth values must always be positive!"

    # Backward pass to check gradient flow
    print("[2/2] Testing the backward pass...")
    loss = output.sum()
    loss.backward()

    # Check whether the first convolutional layer receives gradients
    first_conv_grad = model.inc.conv[0].weight.grad
    assert first_conv_grad is not None, "Error: Gradients did not propagate back to the first layer!"
    print("  -> Backward pass successful; gradients flow through the entire network!")

    print("\n>>> CONGRATULATIONS: THE U-NET MODEL IS READY FOR TRAINING! <<<")


if __name__ == "__main__":
    test_unet()