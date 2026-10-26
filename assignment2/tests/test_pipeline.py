"""End-to-end run of the Trainer on random tensors (no dataset needed). Run: python tests/test_pipeline.py"""

import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.engine.loss import MaskedSILogLoss
from src.engine.trainer import Trainer
from src.models.unet import UNet


class RandomDepth(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        mask = (torch.rand(1, 64, 96, generator=g) > 0.15).float()
        return {"rgb": torch.randn(3, 64, 96, generator=g),
                "depth": (torch.rand(1, 64, 96, generator=g) * 9 + 1) * mask,
                "mask": mask, "domain": "indoor" if i % 2 == 0 else "outdoor"}


def test_trainer_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"train": {"amp": True, "time_budget_min": None}, "out_dir": tmp}
        loaders = {"train": DataLoader(RandomDepth(), batch_size=4), "val": DataLoader(RandomDepth(), batch_size=4)}
        model = UNet()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        summary = Trainer(model, "metric", MaskedSILogLoss(), opt, None, loaders, cfg, device).fit(2)
        assert summary["epochs_run"] == 2
        for f in ("history.csv", "best.pth"):
            assert os.path.exists(os.path.join(tmp, f)), f
    print("pipeline ok", summary)


if __name__ == "__main__":
    test_trainer_runs()
