import json
import os
import platform
import random

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def set_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # needed by cuBLAS when deterministic algorithms are on
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # warn_only: bilinear upsample backward (used in MiDaS) has no deterministic CUDA kernel
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path):
    """Read an experiment yaml on top of configs/base.yaml."""
    with open(os.path.join(ROOT, "configs", "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    resolved = path if os.path.exists(path) else os.path.join(ROOT, path)
    with open(resolved) as f:
        exp = yaml.safe_load(f)
    for k, v in exp.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["out_dir"] = os.path.join(ROOT, cfg["output_root"], cfg["name"])
    cfg["data"]["root"] = os.path.join(ROOT, cfg["data"]["root"])
    return cfg


def env_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
