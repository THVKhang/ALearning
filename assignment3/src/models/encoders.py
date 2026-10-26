import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModalityEncoder(nn.Module):
    """4-stage spatial ConvNet encoder producing both intermediate spatial tokens `(B, C, H', W')`
    and a pooled vector embedding `(B, embed_dim)`.
    """

    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.stem = ConvBlock(in_channels, 32, pool=True)    # 72x96 -> 36x48
        self.stage2 = ConvBlock(32, 64, pool=True)           # 36x48 -> 18x24
        self.stage3 = ConvBlock(64, 96, pool=True)           # 18x24 -> 9x12
        self.stage4 = ConvBlock(96, embed_dim, pool=False)   # 9x12 spatial feature map
        self.SpatialPool = nn.AvgPool2d(kernel_size=3, stride=3)  # exact 9x12 -> 3x4 = 12 spatial region tokens
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x):
        f = self.stem(x)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.stage4(f)
        tokens_2d = self.SpatialPool(f)                      # (B, D, 3, 4)
        b, d, h, w = tokens_2d.shape
        tokens = self.norm(tokens_2d.flatten(2).transpose(1, 2))  # (B, 12, D)
        pooled = tokens.mean(dim=1)                          # (B, D)
        return tokens, pooled

    def forward(self, x):
        _, pooled = self.forward_features(x)
        return pooled
