import torch
import torch.nn as nn

MIN_DEPTH = 1e-3


class MaskedSILogLoss(nn.Module):
    """SILog loss (Eigen et al., 2014) on valid pixels, for the metric-depth U-Net."""

    def __init__(self, lambda_factor=0.5):
        super().__init__()
        self.lambda_factor = lambda_factor

    def forward(self, pred, target, mask):
        valid = (mask > 0) & (target > MIN_DEPTH) & torch.isfinite(target)
        if not valid.any():
            return pred.sum() * 0.0
        # log in fp32, fp16 loses too much precision here
        d = torch.log(pred[valid].float().clamp(min=MIN_DEPTH)) - torch.log(target[valid].float())
        return (d ** 2).mean() - self.lambda_factor * d.mean() ** 2


class ScaleShiftInvariantLoss(nn.Module):
    """MiDaS-style SSI loss (Ranftl et al., 2020): MiDaS predicts relative inverse depth,
    so pred and 1/gt are each normalised per image (median, mean abs deviation) and compared with L1."""

    def forward(self, pred_disp, target_depth, mask):
        losses = []
        for p, t, m in zip(pred_disp, target_depth, mask):
            valid = (m > 0) & (t > MIN_DEPTH) & torch.isfinite(t)
            if valid.sum() < 10:
                continue
            losses.append((self._norm(p[valid].float()) - self._norm(1.0 / t[valid].float())).abs().mean())
        if not losses:
            return pred_disp.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _norm(x):
        med = x.median()
        scale = (x - med).abs().mean().clamp(min=1e-6)
        return (x - med) / scale