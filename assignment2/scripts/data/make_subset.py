import argparse
import csv
import os
import random

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

TARGET_W, TARGET_H = 384, 288           # from 1024x768, about one seventh of the pixels
SPLITS = ("train", "val", "test")
BUDGET = {
    "indoor": {"target": 4000, "max_per_scene": 700},
    "outdoor": {"target": 3000, "max_per_scene": 400},
}


def collect_pairs(domain_path, domain_name):
    """Every RGB file under domain/scene/scan that has both a depth map and a mask."""
    pairs = []
    for scene in sorted(os.listdir(domain_path)):
        scene_path = os.path.join(domain_path, scene)
        if not os.path.isdir(scene_path):
            continue
        for scan in sorted(os.listdir(scene_path)):
            scan_path = os.path.join(scene_path, scan)
            if not os.path.isdir(scan_path):
                continue
            files = set(os.listdir(scan_path))
            for rgb in sorted(f for f in files if f.endswith(".png")):
                base = rgb[:-4]
                if base + "_depth.npy" in files and base + "_depth_mask.npy" in files:
                    pairs.append({
                        "rgb": os.path.join(scan_path, rgb),
                        "depth": os.path.join(scan_path, base + "_depth.npy"),
                        "mask": os.path.join(scan_path, base + "_depth_mask.npy"),
                        "domain": domain_name, "scene": scene, "scan": scan,
                    })
    return pairs


def sample_by_scene_capped(pairs, target_count, max_per_scene):
    """Sample images scene by scene with max cap per scene."""
    scenes = sorted({p["scene"] for p in pairs})
    random.shuffle(scenes)
    selected, used = [], []
    for scene in scenes:
        if len(selected) >= target_count:
            break
        scene_pairs = [p for p in pairs if p["scene"] == scene]
        random.shuffle(scene_pairs)
        selected.extend(scene_pairs[:max_per_scene])
        used.append(scene)
    return selected, used


def split_scenes(scenes_list, train_ratio=0.7, val_ratio=0.15):
    """Split dataset at scene level to prevent data leakage."""
    scenes = list(scenes_list)
    random.shuffle(scenes)
    n = len(scenes)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    return scenes[:n_train], scenes[n_train:n_train + n_val], scenes[n_train + n_val:]


def resize_nearest(arr, out_dtype):
    if cv2 is not None:
        return cv2.resize(arr, (TARGET_W, TARGET_H),
                          interpolation=cv2.INTER_NEAREST).astype(out_dtype)
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(im.resize((TARGET_W, TARGET_H), Image.NEAREST)).astype(out_dtype)


def write_sample(pair, split, out_dir, jpeg_quality):
    base = os.path.basename(pair["rgb"])[:-4]
    folder = os.path.join(out_dir, split)

    # Save RGB as JPEG
    img = Image.open(pair["rgb"]).convert("RGB").resize((TARGET_W, TARGET_H), Image.BILINEAR)
    img.save(os.path.join(folder, base + ".jpg"), quality=jpeg_quality)

    depth = np.load(pair["depth"]).squeeze()
    np.save(os.path.join(folder, base + "_depth.npy"), resize_nearest(depth, np.float16))

    mask = np.load(pair["mask"]).squeeze().astype(np.uint8)
    np.save(os.path.join(folder, base + "_depth_mask.npy"), resize_nearest(mask, np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="upstream DIODE root, the folder containing train/")
    ap.add_argument("--out", default="data/diode_subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--manifest-dir", default="proposal",
                    help="where scenes_used.csv and subset_metadata.csv are written")
    args = ap.parse_args()

    train_root = os.path.join(args.source, "train")
    domains = os.listdir(train_root)
    indoor = next(d for d in domains if "indoor" in d.lower())
    outdoor = next(d for d in domains if "outdoor" in d.lower())

    pairs = {"indoor": collect_pairs(os.path.join(train_root, indoor), "indoor"),
             "outdoor": collect_pairs(os.path.join(train_root, outdoor), "outdoor")}
    for dom in ("indoor", "outdoor"):
        print("%-8s %6d pairs across %2d scenes"
              % (dom, len(pairs[dom]), len({p["scene"] for p in pairs[dom]})))

    random.seed(args.seed)
    subset, used = {}, {}
    for dom in ("indoor", "outdoor"):
        subset[dom], used[dom] = sample_by_scene_capped(
            pairs[dom], BUDGET[dom]["target"], BUDGET[dom]["max_per_scene"])

    random.seed(args.seed)
    assign = {}
    for dom in ("indoor", "outdoor"):
        tr, va, te = split_scenes(used[dom])
        for split, scenes in (("train", tr), ("val", va), ("test", te)):
            for s in scenes:
                assign[s] = split

    everything = subset["indoor"] + subset["outdoor"]
    for p in everything:
        p["split"] = assign[p["scene"]]

    for split in SPLITS:
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
    for i, p in enumerate(everything):
        write_sample(p, p["split"], args.out, args.jpeg_quality)
        if i % 1000 == 0:
            print("  %d/%d" % (i, len(everything)))

    os.makedirs(args.manifest_dir, exist_ok=True)
    rows = {}
    for p in everything:
        key = (p["scene"], p["split"], p["domain"])
        rows[key] = rows.get(key, 0) + 1
    manifest = [{"scene": s, "split": sp, "domain": d, "samples": n}
                for (s, sp, d), n in sorted(rows.items())]
    with open(os.path.join(args.manifest_dir, "scenes_used.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("scene", "split", "domain", "samples"))
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(args.manifest_dir, "subset_metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("rgb", "depth", "mask", "domain", "scene", "scan",
                                          "split"))
        w.writeheader()
        w.writerows(everything)

    print("\n%d scenes, %d images" % (len(manifest), len(everything)))
    for split in SPLITS:
        picked = [r for r in manifest if r["split"] == split]
        ind = sum(r["samples"] for r in picked if r["domain"] == "indoor")
        out = sum(r["samples"] for r in picked if r["domain"] == "outdoor")
        print("  %-5s %2d scenes %5d images (indoor %4d / outdoor %4d)"
              % (split, len(picked), ind + out, ind, out))


if __name__ == "__main__":
    main()
