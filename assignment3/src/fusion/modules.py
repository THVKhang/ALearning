import torch
import torch.nn as nn


class SimpleConcatFusion(nn.Module):
    """Simple late fusion baseline: concatenate unimodal embeddings `[z_rgb ; z_depth]` -> MLP."""

    def __init__(self, embed_dim=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_rgb, z_depth):
        return self.proj(torch.cat([z_rgb, z_depth], dim=-1))


class BidirectionalCrossModalAttention(nn.Module):
    """Intermediate token-level cross-attention between RGB region tokens and Depth geometry tokens,
    with learnable residual gates and LiDAR validity scaling so noisy/missing depth never corrupts RGB.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 12, embed_dim) * 0.02)
        self.rgb_from_depth = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.depth_from_rgb = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.alpha_rgb = nn.Parameter(torch.tensor(0.15))
        self.alpha_depth = nn.Parameter(torch.tensor(0.15))
        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_depth = nn.LayerNorm(embed_dim)

    def forward(self, rgb_tokens, depth_tokens, z_rgb_raw, z_depth_raw, valid_ratio):
        r = rgb_tokens + self.pos_embed
        d = depth_tokens + self.pos_embed
        r_enh, _ = self.rgb_from_depth(query=r, key=d, value=d)
        d_enh, _ = self.depth_from_rgb(query=d, key=r, value=r)
        # Scale depth->rgb cross-attention by valid_ratio so missing depth (valid_ratio=0) leaves RGB untouched
        v_scale = valid_ratio.clamp(0.0, 1.0)
        r_out = self.norm_rgb(z_rgb_raw + self.alpha_rgb * v_scale * r_enh.mean(dim=1))
        d_out = self.norm_depth(z_depth_raw + self.alpha_depth * d_enh.mean(dim=1))
        return r_out, d_out


class ValidityConditionedGatedFusion(nn.Module):
    """Reliability-gated fusion conditioned on both modality embeddings and LiDAR valid-pixel ratio `r_valid`."""

    def __init__(self, embed_dim=128, use_validity_gate=True, dropout=0.2):
        super().__init__()
        self.use_validity_gate = use_validity_gate
        if use_validity_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2 + 1, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim),
                nn.Sigmoid(),
            )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_rgb, z_depth, valid_ratio):
        v_scale = valid_ratio.clamp(0.0, 1.0)
        if self.use_validity_gate:
            g = self.gate_net(torch.cat([z_rgb, z_depth * v_scale, v_scale], dim=-1))
            r_weighted = g * z_rgb
            d_weighted = (1.0 - g) * v_scale * z_depth
            interaction = r_weighted + d_weighted
        else:
            g = torch.full_like(z_rgb, 0.5)
            r_weighted = 0.5 * z_rgb
            d_weighted = 0.5 * z_depth
            interaction = r_weighted + d_weighted
        fused = self.fuse_mlp(torch.cat([r_weighted, d_weighted, interaction], dim=-1))
        return fused, g
