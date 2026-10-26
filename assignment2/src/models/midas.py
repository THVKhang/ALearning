import torch
import torch.nn as nn


class MiDaSSmall(nn.Module):
    """MiDaS_small from torch.hub. Output is relative inverse depth (B, 1, H, W)."""

    def __init__(self, freeze_encoder=False):
        super().__init__()
        # MiDaS_small loads its EfficientNet backbone from this repo, trust it first
        torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_lite3",
                       pretrained=False, trust_repo=True, skip_validation=True)
        self.net = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.net.pretrained.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # frozen encoder also keeps its BatchNorm running stats fixed
        if self.freeze_encoder:
            self.net.pretrained.eval()
        return self

    def forward(self, x):
        return self.net(x).unsqueeze(1)
