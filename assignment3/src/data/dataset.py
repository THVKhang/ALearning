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
MIN_DEPTH = 0.6
MAX_DEPTH = 200.0
LOG_MAX = float(np.log1p(MAX_DEPTH))


def parse_sample_id(sid):
    """Parse `<scene>_<scan>_<domain>_<h_angle>_<v_angle>` into metadata and 6-class label.

    Classes (6):
      0: Indoor  - Level (000-010 deg)
      1: Indoor  - Mid-Elevated (020-030 deg)
      2: Indoor  - High-Elevated (040-050 deg)
      3: Outdoor - Level (000-010 deg)
      4: Outdoor - Mid-Elevated (020-030 deg)
      5: Outdoor - High-Elevated (040-050 deg)
    """
    scene, scan, dom, h_str, v_str = sid.split("_")
    v_angle = int(v_str)
    dom_idx = 0 if dom == "indoors" else 3
    if v_angle <= 10:
        elev_idx = 0
    elif v_angle <= 30:
        elev_idx = 1
    else:
        elev_idx = 2
    return {
        "id": sid,
        "scene": scene,
        "scan": scan,
        "domain": "indoor" if dom == "indoors" else "outdoor",
        "h_angle": int(h_str),
        "v_angle": v_angle,
        "label": dom_idx + elev_idx,
    }


def list_paired_samples(split_dir):
    """Verify cross-modal pair completeness: every RGB `.jpg` must have matching `_depth.npy` and `_depth_mask.npy`."""
    files = set(os.listdir(split_dir))
    rgb_ids = sorted(f[:-4] for f in files if f.endswith(".jpg"))
    depth_ids = {f[: -len(DEPTH_SUFFIX)] for f in files if f.endswith(DEPTH_SUFFIX)}
    mask_ids = {f[: -len(MASK_SUFFIX)] for f in files if f.endswith(MASK_SUFFIX)}
    paired = [i for i in rgb_ids if i in depth_ids and i in mask_ids]
    if len(paired) != len(rgb_ids):
        raise RuntimeError(f"{split_dir}: {len(rgb_ids) - len(paired)} unpaired samples detected")
    return paired


def _load_and_downsample_split(split_dir, img_size=(72, 96)):
    """Load paired RGB and LiDAR Depth+Mask arrays and downsample to `img_size=(H, W)`."""
    th, tw = img_size
    ids = list_paired_samples(split_dir)
    n = len(ids)
    rgbs = np.empty((n, 3, th, tw), dtype=np.uint8)
    depths = np.empty((n, th, tw), dtype=np.float16)
    masks = np.empty((n, th, tw), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    valid_ratios = np.empty(n, dtype=np.float32)

    for idx, sid in enumerate(ids):
        meta = parse_sample_id(sid)
        base = os.path.join(split_dir, sid)
        img = Image.open(base + ".jpg").convert("RGB")
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.BILINEAR)
        rgb_arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

        d_arr = np.load(base + DEPTH_SUFFIX).astype(np.float32).squeeze()
        m_arr = np.load(base + MASK_SUFFIX).squeeze()
        sh, sw = d_arr.shape
        if (sh, sw) != (th, tw):
            ys = np.linspace(0, sh - 1, th).astype(np.int64)
            xs = np.linspace(0, sw - 1, tw).astype(np.int64)
            d_arr = d_arr[np.ix_(ys, xs)]
            m_arr = m_arr[np.ix_(ys, xs)]

        max_d = 50.0 if meta["domain"] == "indoor" else 200.0
        valid = (m_arr > 0) & np.isfinite(d_arr) & (d_arr >= MIN_DEPTH) & (d_arr <= max_d)
        d_clean = np.where(valid, d_arr, 0.0).astype(np.float16)

        rgbs[idx] = rgb_arr
        depths[idx] = d_clean
        masks[idx] = valid.astype(np.uint8)
        labels[idx] = meta["label"]
        valid_ratios[idx] = float(valid.mean())

    return {
        "ids": ids,
        "rgb": rgbs,
        "depth": depths,
        "mask": masks,
        "label": labels,
        "valid_ratio": valid_ratios,
    }


def load_cached_split(root, split, img_size=(72, 96)):
    """Cache downsampled paired tensors under `assignment3/data/cache_<H>x<W>_<split>.pt` for fast multi-model training."""
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cache_{img_size[0]}x{img_size[1]}_{split}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    split_dir = os.path.join(root, split)
    data = _load_and_downsample_split(split_dir, img_size=img_size)
    torch.save(data, cache_path)
    return data


class DiodeMultimodalDataset(Dataset):
    """Paired RGB (Modality A) + 3D LiDAR Depth/Validity/Gradient (Modality B) dataset.

    Modality A (`rgb`): shape `(3, H, W)` ImageNet-normalized RGB appearance tensor.
    Modality B (`depth_geo`): shape `(3, H, W)` geometric tensor:
      - Channel 0: normalized log-depth `log(1 + d) / log(1 + 200)` on valid pixels (0 where invalid).
      - Channel 1: binary validity mask `m in {0, 1}` indicating valid laser returns vs sky/dropout.
      - Channel 2: vertical log-depth finite difference `d(log_depth)/dy` on valid pairs, capturing surface tilt.
    """

    def __init__(self, root, split="train", img_size=(72, 96), hflip=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.hflip = hflip
        data = load_cached_split(root, split, img_size=tuple(img_size))
        self.ids = data["ids"]
        self.rgbs = data["rgb"]
        self.depths = data["depth"]
        self.masks = data["mask"]
        self.labels = data["label"]
        self.valid_ratios = data["valid_ratio"]

        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rgb = torch.from_numpy(self.rgbs[idx].astype(np.float32) / 255.0)
        depth = torch.from_numpy(self.depths[idx].astype(np.float32))
        mask = torch.from_numpy(self.masks[idx].astype(np.float32))

        # Synchronous horizontal flip preserves both domain and vertical camera elevation label
        if self.hflip and torch.rand(1).item() < 0.5:
            rgb = rgb.flip(-1)
            depth = depth.flip(-1)
            mask = mask.flip(-1)

        rgb = (rgb - self.mean) / self.std

        log_d = torch.log1p(depth.clamp(min=0.0)) / LOG_MAX
        log_d = log_d * mask

        # Vertical depth gradient (bottom-to-top surface slant cue)
        grad_y = torch.zeros_like(log_d)
        both_valid = (mask[1:, :] > 0) & (mask[:-1, :] > 0)
        grad_y[1:, :] = torch.where(both_valid, (log_d[1:, :] - log_d[:-1, :]) * 10.0, torch.zeros_like(log_d[1:, :]))
        grad_y = grad_y.clamp(-2.0, 2.0)

        depth_geo = torch.stack([log_d, mask, grad_y], dim=0)

        return {
            "rgb": rgb,
            "depth_geo": depth_geo,
            "valid_ratio": torch.tensor([self.valid_ratios[idx]], dtype=torch.float32),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "id": self.ids[idx],
        }


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def get_dataloaders(root, img_size=(72, 96), batch_size=64, num_workers=0, augment=True, seed=42, splits=SPLITS):
    gen = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in splits:
        train = split == "train"
        ds = DiodeMultimodalDataset(root, split=split, img_size=img_size, hflip=augment and train)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=gen if train else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
    return loaders
