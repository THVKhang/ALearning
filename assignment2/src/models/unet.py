import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    Lightweight U-Net for Monocular Depth Estimation.
    Input:  (B, 3, H, W) - RGB Image
    Output: (B, 1, H, W) - Depth map in metres
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, max_depth: float = 200.0):
        super().__init__()
        self.max_depth = max_depth

        # --- ENCODER (Trích xuất đặc trưng không gian) ---
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # --- BOTTLENECK (Tầng sâu nhất, ngữ nghĩa cao nhất) ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- DECODER (Phóng to và nối Skip Connections) ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)  # 256 (từ up) + 256 (skip) = 512

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)  # 128 (từ up) + 128 (skip) = 256

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # 64 (từ up) + 64 (skip) = 128

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)    # 32 (từ up) + 32 (skip) = 64

        # --- OUTPUT HEAD ---
        # Đưa về 1 kênh bản đồ độ sâu
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Nhánh Encoder đi xuống
        x1 = self.inc(x)        # Kích thước: (B, 32, H, W)
        x2 = self.down1(x1)     # Kích thước: (B, 64, H/2, W/2)
        x3 = self.down2(x2)     # Kích thước: (B, 128, H/4, W/4)
        x4 = self.down3(x3)     # Kích thước: (B, 256, H/8, W/8)

        # 2. Tầng Bottleneck
        b = self.bottleneck(x4) # Kích thước: (B, 512, H/16, W/16)

        # 3. Nhánh Decoder đi lên kèm Skip Connections
        d1 = self.up1(b)
        d1 = self.conv_up1(torch.cat([d1, x4], dim=1))

        d2 = self.up2(d1)
        d2 = self.conv_up2(torch.cat([d2, x3], dim=1))

        d3 = self.up3(d2)
        d3 = self.conv_up3(torch.cat([d3, x2], dim=1))

        d4 = self.up4(d3)
        d4 = self.conv_up4(torch.cat([d4, x1], dim=1))

        # 4. Tầng Output đưa về độ sâu vật lý
        # Dùng F.relu + 1e-3 để đảm bảo độ sâu luôn là số dương thực tế
        out = self.outc(d4)
        depth = F.relu(out) + 1e-3
        return depth