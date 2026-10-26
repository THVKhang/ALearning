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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path):
    """Load an experiment YAML on top of configs/base.yaml."""
    with open(os.path.join(ROOT, "configs", "base.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resolved = path if os.path.exists(path) else os.path.join(ROOT, path)
    with open(resolved, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    for k, v in exp.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["out_dir"] = os.path.join(ROOT, cfg["output_root"], cfg["name"])
    if not os.path.isabs(cfg["data"]["root"]):
        cfg["data"]["root"] = os.path.abspath(os.path.join(ROOT, cfg["data"]["root"]))
    return cfg


def env_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
