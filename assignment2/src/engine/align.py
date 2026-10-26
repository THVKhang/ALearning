import torch

from src.data.dataset import MAX_DEPTH
from src.engine.metrics import MIN_DEPTH, valid_mask


def fit_scale_shift(pred, target, valid):
    """Least squares s, t so that s * pred + t ~ target on valid pixels."""
    p = pred[valid].double()
    y = target[valid].double()
    n = p.numel()
    sp, sy, spp, spy = p.sum(), y.sum(), (p * p).sum(), (p * y).sum()
    det = n * spp - sp * sp
    if n < 2 or det.abs() < 1e-12:
        return 1.0, 0.0
    s = (n * spy - sp * sy) / det
    t = (spp * sy - sp * spy) / det
    return s.item(), t.item()


def align_disparity(pred_disp, gt, mask, max_depth):
    """MiDaS outputs relative inverse depth. Fit scale/shift in disparity space against
    1/gt on valid pixels of this image, then invert back to metres."""
    valid = valid_mask(gt, mask, max_depth)
    if not valid.any():
        return torch.full_like(gt, MIN_DEPTH), (1.0, 0.0)
    target = torch.zeros_like(gt)
    target[valid] = 1.0 / gt[valid]
    s, t = fit_scale_shift(pred_disp.float(), target, valid)
    disp = (s * pred_disp.float() + t).clamp(1.0 / max_depth, 1.0 / MIN_DEPTH)
    return 1.0 / disp, (s, t)


def align_batch(pred_disp, gt, mask, domains):
    out = torch.empty_like(gt)
    for i, dom in enumerate(domains):
        out[i], _ = align_disparity(pred_disp[i], gt[i], mask[i], MAX_DEPTH[dom])
    return out


def depth_to_disp(depth):
    return 1.0 / depth.float().clamp(min=MIN_DEPTH)
