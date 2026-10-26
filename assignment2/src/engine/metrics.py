import math

import torch

from src.data.dataset import MAX_DEPTH, MIN_DEPTH as GT_MIN

MIN_DEPTH = 1e-3  # lower clamp for predictions
# distance bins for error analysis (metres)
BINS = {"near (<5m)": (0.0, 5.0), "mid (5-20m)": (5.0, 20.0), "far (>20m)": (20.0, float("inf"))}


def valid_mask(gt, mask, max_depth):
    return (mask > 0) & torch.isfinite(gt) & (gt >= GT_MIN) & (gt <= max_depth)


def depth_metrics(pred, gt, mask, max_depth):
    """Abs Rel, RMSE, delta1 on valid pixels of one image. pred/gt in metres.

    Bad predictions are clamped into [MIN_DEPTH, max_depth] instead of being dropped,
    otherwise a model that outputs 0 would get its worst pixels removed from the score.
    """
    valid = valid_mask(gt, mask, max_depth)
    if not valid.any():
        return None
    t = gt[valid].float()
    p = torch.nan_to_num(pred[valid].float(), nan=MIN_DEPTH, posinf=max_depth, neginf=MIN_DEPTH)
    p = p.clamp(MIN_DEPTH, max_depth)
    ratio = torch.maximum(p / t, t / p)
    return {
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse": torch.sqrt(((p - t) ** 2).mean()).item(),
        "delta1": (ratio < 1.25).float().mean().item(),
    }


class MetricTracker:
    """Per-image metrics averaged over images, split into indoor / outdoor / overall.
    Also keeps pixel-pooled errors per distance bin for the error analysis."""

    KEYS = ("abs_rel", "rmse", "delta1")

    def __init__(self):
        self.rows = {"indoor": [], "outdoor": []}
        self.bins = {d: {b: [0.0, 0.0, 0.0, 0] for b in BINS} for d in ("indoor", "outdoor")}

    @torch.no_grad()
    def update(self, pred, gt, mask, domains):
        for i, dom in enumerate(domains):
            max_d = MAX_DEPTH[dom]
            m = depth_metrics(pred[i], gt[i], mask[i], max_d)
            if m is None:
                continue
            self.rows[dom].append(m)

            valid = valid_mask(gt[i], mask[i], max_d)
            t = gt[i][valid].float()
            p = torch.nan_to_num(pred[i][valid].float(), nan=MIN_DEPTH).clamp(MIN_DEPTH, max_d)
            for b, (lo, hi) in BINS.items():
                sel = (t >= lo) & (t < hi)
                if sel.any():
                    acc = self.bins[dom][b]
                    acc[0] += ((p[sel] - t[sel]).abs() / t[sel]).sum().item()
                    acc[1] += ((p[sel] - t[sel]) ** 2).sum().item()
                    acc[2] += (torch.maximum(p[sel] / t[sel], t[sel] / p[sel]) < 1.25).sum().item()
                    acc[3] += int(sel.sum())

    def _mean(self, rows):
        if not rows:
            return {k: float("nan") for k in self.KEYS} | {"n_images": 0}
        out = {k: sum(r[k] for r in rows) / len(rows) for k in self.KEYS}
        out["n_images"] = len(rows)
        return out

    def summary(self):
        res = {
            "indoor": self._mean(self.rows["indoor"]),
            "outdoor": self._mean(self.rows["outdoor"]),
            "overall": self._mean(self.rows["indoor"] + self.rows["outdoor"]),
        }
        res["bins"] = {
            dom: {b: ({"abs_rel": a / n, "rmse": math.sqrt(s / n), "delta1": d / n, "n_pixels": n}
                      if n else None)
                  for b, (a, s, d, n) in self.bins[dom].items()}
            for dom in self.bins
        }
        return res