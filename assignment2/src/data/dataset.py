import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_SUFFIX = "_depth.npy"
MASK_SUFFIX = "_depth_mask.npy"
# from train EDA: above this is < 0.01% of valid pixels, treated as noise
MAX_DEPTH = {"indoor": 50.0, "outdoor": 200.0}


def list_samples(split_dir):
    """Get sample ids that have all 3 files (rgb, depth, mask)."""
    files = set(os.listdir(split_dir))
    ids = sorted(f[:-4] for f in files if f.endswith(".jpg"))
    complete = [i for i in ids if i + DEPTH_SUFFIX in files and i + MASK_SUFFIX in files]
    if len(complete) != len(ids):
        raise RuntimeError(f"{split_dir}: {len(ids) - len(complete)} samples missing depth/mask")
    return complete


def domain_of(sample_id):
    return "indoor" if "_indoors_" in sample_id else "outdoor"


class DiodeDepthDataset(Dataset):
    """Returns rgb (3,H,W), depth (1,H,W) in metres, mask (1,H,W) with 0/1."""

    def __init__(self, root, split="train", normalize=True, hflip=False, max_depth=MAX_DEPTH):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.dir = os.path.join(root, split)
        self.split = split
        self.ids = list_samples(self.dir)
        self.normalize = normalize
        self.hflip = hflip
        self.max_depth = max_depth

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        base = os.path.join(self.dir, sid)

        rgb = np.asarray(Image.open(base + ".jpg").convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(base + DEPTH_SUFFIX).astype(np.float32)
        mask = np.load(base + MASK_SUFFIX)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if mask.ndim == 3:
            mask = mask[..., 0]

        # bad pixels (mask 0, nan, depth <= 0, too far) -> depth 0
        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        if self.max_depth:
            valid &= depth <= self.max_depth[domain_of(sid)]
        depth = np.where(valid, depth, 0.0).astype(np.float32)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)

        if self.hflip and torch.rand(1).item() < 0.5:
            rgb, depth, mask = rgb.flip(-1), depth.flip(-1), mask.flip(-1)
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            rgb = (rgb - mean) / std

        return {"rgb": rgb, "depth": depth, "mask": mask, "id": sid, "domain": domain_of(sid)}


def get_dataloaders(root, batch_size=8, num_workers=2, augment=False, seed=42):
    """Make train/val/test loaders (split by scene was done on Kaggle)."""
    gen = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in SPLITS:
        ds = DiodeDepthDataset(root, split, hflip=augment and split == "train")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), generator=gen if split == "train" else None,
        )
    return loaders
