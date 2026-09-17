"""Fashion-MNIST / MNIST loading with a seeded train/validation/test split."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

INPUT_SHAPE = (1, 28, 28)

DATASETS = {
    "fashion_mnist": {
        "cls": FashionMNIST,
        "mean": 0.2860,
        "std": 0.3530,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
    "mnist": {
        "cls": MNIST,
        "mean": 0.1307,
        "std": 0.3081,
        "classes": [str(digit) for digit in range(10)],
    },
}


def build_transform(spec, train, augment):
    ops = []
    if train and augment:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(28, padding=2)]
    ops += [transforms.ToTensor(), transforms.Normalize((spec["mean"],), (spec["std"],))]
    return transforms.Compose(ops)


def split_indices(targets, val_fraction, seed, stratified=True):
    """Return (train_idx, val_idx); stratified keeps each class ratio in both parts."""
    rng = np.random.default_rng(seed)
    if not stratified:
        perm = rng.permutation(len(targets))
        cut = int(round(len(targets) * val_fraction))
        return np.sort(perm[cut:]), np.sort(perm[:cut])

    train_parts, val_parts = [], []
    for label in np.unique(targets):
        idx = np.flatnonzero(targets == label)
        rng.shuffle(idx)
        cut = int(round(len(idx) * val_fraction))
        val_parts.append(idx[:cut])
        train_parts.append(idx[cut:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def get_dataloaders(dataset="fashion_mnist", data_root="data", val_fraction=0.1,
                    stratified=True, batch_size=128, num_workers=0, augment=False,
                    seed=42, download=True):
    """Build train/val/test loaders plus an `info` dict of split facts for the report."""
    spec = DATASETS[dataset]
    ds_cls = spec["cls"]

    # Two views of the same 60k training set: the train subset may be augmented,
    # the validation subset never is.
    train_source = ds_cls(data_root, train=True, download=download,
                          transform=build_transform(spec, train=True, augment=augment))
    val_source = ds_cls(data_root, train=True, download=download,
                        transform=build_transform(spec, train=False, augment=False))
    test_set = ds_cls(data_root, train=False, download=download,
                      transform=build_transform(spec, train=False, augment=False))

    targets = train_source.targets.numpy()
    train_idx, val_idx = split_indices(targets, val_fraction, seed, stratified)

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(Subset(train_source, train_idx), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, generator=generator,
                            pin_memory=pin_memory),
        "val": DataLoader(Subset(val_source, val_idx), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }

    num_classes = len(spec["classes"])
    test_targets = test_set.targets.numpy()
    info = {
        "dataset": dataset,
        "classes": spec["classes"],
        "num_classes": num_classes,
        "input_shape": list(INPUT_SHAPE),
        "normalize": {"mean": spec["mean"], "std": spec["std"]},
        "augment": bool(augment),
        "stratified": bool(stratified),
        "val_fraction": val_fraction,
        "seed": seed,
        "batch_size": batch_size,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_targets)),
        },
        "class_counts": {
            "train": np.bincount(targets[train_idx], minlength=num_classes).tolist(),
            "val": np.bincount(targets[val_idx], minlength=num_classes).tolist(),
            "test": np.bincount(test_targets, minlength=num_classes).tolist(),
        },
    }
    return loaders, info
