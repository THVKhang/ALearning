import torch
import torch.nn as nn

from src.fusion.modules import (
    BidirectionalCrossModalAttention,
    SimpleConcatFusion,
    ValidityConditionedGatedFusion,
)
from src.models.encoders import ModalityEncoder


class UnimodalClassifier(nn.Module):
    """Unimodal baseline using either Modality A (`rgb`) or Modality B (`depth_geo`) only."""

    def __init__(self, modality="rgb", embed_dim=128, num_classes=6, dropout=0.2):
        super().__init__()
        assert modality in ("rgb", "depth")
        self.modality = modality
        self.encoder = ModalityEncoder(in_channels=3, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, batch):
        x = batch["rgb"] if self.modality == "rgb" else batch["depth_geo"]
        z = self.encoder(x)
        return {"logits": self.head(z)}


class SimpleFusionClassifier(nn.Module):
    """Simple late-fusion baseline: independent RGB & Depth encoders + concatenation MLP."""

    def __init__(self, embed_dim=128, num_classes=6, dropout=0.2):
        super().__init__()
        self.rgb_encoder = ModalityEncoder(in_channels=3, embed_dim=embed_dim)
        self.depth_encoder = ModalityEncoder(in_channels=3, embed_dim=embed_dim)
        self.fusion = SimpleConcatFusion(embed_dim=embed_dim, dropout=dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, batch):
        z_rgb = self.rgb_encoder(batch["rgb"])
        z_depth = self.depth_encoder(batch["depth_geo"])
        z_fused = self.fusion(z_rgb, z_depth)
        return {"logits": self.head(z_fused)}


class ImprovedMultimodalClassifier(nn.Module):
    """Improved multimodal architecture:
      1. Token-level Bidirectional Cross-Modal Attention across 3x4 spatial regions.
      2. LiDAR Validity-Conditioned Reliability Gated Fusion (`g(z_rgb, z_depth, r_valid)`).
      3. Stochastic Modality Dropout during training to handle missing/degraded LiDAR returns.
      4. Auxiliary per-modality heads (`logits_rgb`, `logits_depth`) to prevent modality laziness.
    """

    def __init__(
        self,
        embed_dim=128,
        num_classes=6,
        use_cross_attn=True,
        use_validity_gate=True,
        modality_dropout=0.15,
        dropout=0.2,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_validity_gate = use_validity_gate
        self.modality_dropout = modality_dropout

        self.rgb_encoder = ModalityEncoder(in_channels=3, embed_dim=embed_dim)
        self.depth_encoder = ModalityEncoder(in_channels=3, embed_dim=embed_dim)

        if use_cross_attn:
            self.cross_attn = BidirectionalCrossModalAttention(embed_dim=embed_dim, num_heads=4)
        self.fusion = ValidityConditionedGatedFusion(
            embed_dim=embed_dim, use_validity_gate=use_validity_gate, dropout=dropout
        )

        self.head_fused = nn.Linear(embed_dim, num_classes)
        self.head_rgb = nn.Linear(embed_dim, num_classes)
        self.head_depth = nn.Linear(embed_dim, num_classes)

    def _apply_modality_dropout(self, depth_geo, valid_ratio):
        if not self.training or self.modality_dropout <= 0.0:
            return depth_geo, valid_ratio
        b = depth_geo.size(0)
        drop_mask = torch.rand(b, 1, 1, 1, device=depth_geo.device) < self.modality_dropout
        depth_out = torch.where(drop_mask, torch.zeros_like(depth_geo), depth_geo)
        ratio_out = torch.where(drop_mask.view(b, 1), torch.zeros_like(valid_ratio), valid_ratio)
        return depth_out, ratio_out

    def forward(self, batch):
        rgb = batch["rgb"]
        depth_geo, valid_ratio = self._apply_modality_dropout(batch["depth_geo"], batch["valid_ratio"])

        rgb_tokens, z_rgb_raw = self.rgb_encoder.forward_features(rgb)
        depth_tokens, z_depth_raw = self.depth_encoder.forward_features(depth_geo)

        if self.use_cross_attn:
            z_rgb, z_depth = self.cross_attn(rgb_tokens, depth_tokens, z_rgb_raw, z_depth_raw, valid_ratio)
        else:
            z_rgb, z_depth = z_rgb_raw, z_depth_raw

        z_fused, gate = self.fusion(z_rgb, z_depth, valid_ratio)
        return {
            "logits": self.head_fused(z_fused),
            "logits_rgb": self.head_rgb(z_rgb_raw),
            "logits_depth": self.head_depth(z_depth_raw),
            "gate_mean": gate.mean(dim=-1),
        }


def build_model(cfg):
    mtype = cfg["model"]
    if mtype == "rgb_only":
        return UnimodalClassifier(modality="rgb")
    if mtype == "depth_only":
        return UnimodalClassifier(modality="depth")
    if mtype == "simple_fusion":
        return SimpleFusionClassifier()
    if mtype == "improved_fusion":
        return ImprovedMultimodalClassifier(
            use_cross_attn=cfg.get("use_cross_attn", True),
            use_validity_gate=cfg.get("use_validity_gate", True),
            modality_dropout=cfg["train"].get("modality_dropout", 0.15),
        )
    raise ValueError(f"Unknown model type: {mtype}")
