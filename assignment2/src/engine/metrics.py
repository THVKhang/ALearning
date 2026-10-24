from typing import Dict, List, Optional, Union
import numpy as np
import torch

def compute_depth_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    min_depth: float = 1e-3,
) -> Dict[str, float]:
    """
    Tính các chỉ số đánh giá Monocular Depth Estimation trên các pixel hợp lệ.
    """
    # 1. Chuyển Tensor sang NumPy để tính toán số học nhanh và ổn định
    if isinstance(pred, torch.Tensor):
        pred_arr = pred.detach().cpu().numpy()
    else:
        pred_arr = np.asarray(pred)

    if isinstance(target, torch.Tensor):
        target_arr = target.detach().cpu().numpy()
    else:
        target_arr = np.asarray(target)

    # 2. Xử lý mask hợp lệ
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_arr = mask.detach().cpu().numpy() > 0
        else:
            mask_arr = np.asarray(mask) > 0
    else:
        mask_arr = np.ones_like(target_arr, dtype=bool)

  # 3. Lọc các pixel: mask == 1, target > 0, pred > 0 và không bị NaN/Inf
    valid = mask_arr & np.isfinite(target_arr) & np.isfinite(pred_arr)
    valid &= (target_arr > min_depth) & (pred_arr > min_depth)

    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "delta_1": float("nan"),
            "delta_2": float("nan"),
            "delta_3": float("nan"),
            "n_valid_pixels": 0,
        }

    # Lấy mảng 1D các pixel hợp lệ
    p = pred_arr[valid]
    t = target_arr[valid]


    # 4. Tính các chỉ số đánh giá    # 4. Tính Abs Rel: mean(|p - t| / t)
    abs_diff = np.abs(p - t)
    abs_rel = float(np.mean(abs_diff / t))

    # 5. Tính RMSE: sqrt(mean((p - t)^2))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    # 6. Tính Threshold Accuracy (delta): ratio = max(p / t, t / p)
    ratio = np.maximum(p / t, t / p)
    delta_1 = float(np.mean(ratio < 1.25))           # Ngưỡng 1.25^1 (25%)
    delta_2 = float(np.mean(ratio < 1.25 ** 2))      # Ngưỡng 1.25^2 (56.25%)
    delta_3 = float(np.mean(ratio < 1.25 ** 3))      # Ngưỡng 1.25^3 (95.3%)

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "n_valid_pixels": n_valid,
    }

class DepthMetricTracker:
    """
    Theo dõi và tích lũy metrics qua các batch trong quá trình Validation / Testing.
    Tự động phân tách thống kê cho 'indoor', 'outdoor' và 'overall'.
    """
    def __init__(self, min_depth: float = 1e-3):
        self.min_depth = min_depth
        self.reset()

    def reset(self):
        """Khởi tạo lại bộ đếm khi bắt đầu epoch mới."""
        self.records: List[Dict[str, float]] = []
        self.domain_records: Dict[str, List[Dict[str, float]]] = {
            "indoor": [],
            "outdoor": [],
        }

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        domains: Optional[List[str]] = None,
    ):
        """Cập nhật dữ liệu từ 1 batch."""
        b_size = pred.shape[0]
        for i in range(b_size):
            p_i = pred[i]
            t_i = target[i]
            m_i = mask[i] if mask is not None else None
            dom_i = domains[i] if (domains is not None and i < len(domains)) else None

            # Tính metric cho từng ảnh
            res = compute_depth_metrics(p_i, t_i, m_i, min_depth=self.min_depth)
            if res["n_valid_pixels"] > 0:
                self.records.append(res)
                if dom_i in self.domain_records:
                    self.domain_records[dom_i].append(res)

    def _aggregate(self, records: List[Dict[str, float]]) -> Dict[str, float]:
        """Tính giá trị trung bình từ danh sách records."""
        if not records:
            return {
                "abs_rel": float("nan"),
                "rmse": float("nan"),
                "delta_1": float("nan"),
                "delta_2": float("nan"),
                "delta_3": float("nan"),
                "count": 0,
            }
        keys = ["abs_rel", "rmse", "delta_1", "delta_2", "delta_3"]
        out = {k: float(np.mean([r[k] for r in records if np.isfinite(r[k])])) for k in keys}
        out["count"] = len(records)
        return out

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Trả về kết quả trung bình cho overall, indoor, outdoor."""
        return {
            "overall": self._aggregate(self.records),
            "indoor": self._aggregate(self.domain_records["indoor"]),
            "outdoor": self._aggregate(self.domain_records["outdoor"]),
        }