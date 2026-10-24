from typing import Optional
import torch
import torch.nn as nn

class MaskedSILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss (Eigen et al., NeurIPS 2014)
    Chỉ tính toán trên các pixel hợp lệ (mask > 0).
    """
    def __init__(self, lambda_factor: float = 0.5, min_depth: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        self.lambda_factor = float(lambda_factor)
        self.min_depth = float(min_depth)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Kiểm tra kích thước đồng nhất
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

        # 2. Tạo boolean mask lọc các pixel hợp lệ:
        # Pixel phải có mask > 0, độ sâu > min_depth và không bị NaN/Inf
        if mask is not None:
            valid = (mask > 0) & (target > self.min_depth) & torch.isfinite(target) & torch.isfinite(pred)
        else:
            valid = (target > self.min_depth) & torch.isfinite(target) & torch.isfinite(pred)

        n_valid = valid.sum()
        if n_valid == 0:
            # Trả về 0 nhưng vẫn giữ liên kết với pred để không làm gãy đồ thị đạo hàm
            return pred.sum() * 0.0

        # 3. Lấy ra các pixel hợp lệ
        p = pred[valid].clamp(min=self.min_depth)
        t = target[valid].clamp(min=self.min_depth)

        # 4. Tính sai số trên miền Logarithm
        diff = torch.log(p) - torch.log(t)

        # 5. Áp dụng công thức SILog
        # Thành phần 1: Trung bình bình phương sai số
        loss_sq = torch.mean(diff ** 2)
        # Thành phần 2: Bình phương sai số trung bình
        loss_sum = (torch.mean(diff)) ** 2

        loss = loss_sq - self.lambda_factor * loss_sum
        return loss