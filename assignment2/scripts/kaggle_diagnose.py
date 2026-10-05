# Self-contained version of export_scenes.py + diagnose_subset.py, for a Kaggle notebook.
#
# The subset lives on Kaggle, not on the machine that holds this repository, so the two
# diagnostics cannot import from src/. Paste this whole file into one notebook cell and run
# it. It needs nothing but numpy and pillow, both preinstalled on Kaggle.
#
# It prints the scene manifest and the resize verdict, and writes scenes_used.csv into
# /kaggle/working so it can be downloaded and committed to proposal/.

import os
import csv
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

# Leave ROOT as None to search /kaggle/input for the folder holding train/ val/ test/.
ROOT = None
SPLITS = ("train", "val", "test")
PROBE = 12                      # images per split for the resize probe


def find_root(start="/kaggle/input"):
    if ROOT:
        return ROOT
    for base, dirs, _ in os.walk(start):
        if all(d in dirs for d in SPLITS):
            return base
    raise SystemExit("No folder with train/ val/ test/ under %s. Set ROOT manually." % start)


def list_ids(split_dir):
    files = set(os.listdir(split_dir))
    ids = sorted(f[:-4] for f in files if f.endswith(".jpg"))
    missing = [i for i in ids
               if i + "_depth.npy" not in files or i + "_depth_mask.npy" not in files]
    if missing:
        print("  WARNING %s: %d images have no depth or mask" % (split_dir, len(missing)))
    return [i for i in ids if i not in set(missing)]


def parse_id(sid):
    # 00005_00039_indoors_250_050 -> scene, scan, domain, h angle, v angle
    scene, scan, dom, h, v = sid.split("_")
    return scene, int(scan), "indoor" if dom == "indoors" else "outdoor", int(h), int(v)


def border_ratio(depth, valid):
    """Median depth of valid pixels touching an invalid pixel, over the interior median.

    DIODE stores invalid depth as 0, so any averaging filter mixes those zeros into
    neighbouring valid pixels and drags them down; nearest neighbour copies single pixels
    and leaves them alone. Calibrated on a synthetic 768x1024 map (2 m object on an 8 m
    background, invalid band) downsampled to 384x288: nearest 1.000, bilinear 0.116.
    """
    inv = ~valid
    nb = np.zeros_like(inv)
    nb[1:, :] |= inv[:-1, :]
    nb[:-1, :] |= inv[1:, :]
    nb[:, 1:] |= inv[:, :-1]
    nb[:, :-1] |= inv[:, 1:]
    border, interior = valid & nb, valid & ~nb
    if not border.any() or not interior.any():
        return None
    return float(np.median(depth[border]) / np.median(depth[interior]))


root = find_root()
print("root: %s\n" % root)

by_scene = defaultdict(list)
for split in SPLITS:
    for sid in list_ids(os.path.join(root, split)):
        scene, scan, dom, h, v = parse_id(sid)
        by_scene[(split, scene, dom)].append((scan, h, v))

# ------------------------------------------------- 1. scene manifest and leakage check
rows = []
for (split, scene, dom), items in by_scene.items():
    rows.append({"scene": scene, "split": split, "domain": dom, "samples": len(items)})
rows.sort(key=lambda r: r["scene"])

seen = {}
for r in rows:
    if r["scene"] in seen and seen[r["scene"]] != r["split"]:
        raise SystemExit("LEAKAGE: scene %s in both %s and %s"
                         % (r["scene"], seen[r["scene"]], r["split"]))
    seen[r["scene"]] = r["split"]

out_csv = "/kaggle/working/scenes_used.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=("scene", "split", "domain", "samples"))
    w.writeheader()
    w.writerows(rows)

print("== scenes_used.csv ==")
print("scene,split,domain,samples")
for r in rows:
    print("%s,%s,%s,%d" % (r["scene"], r["split"], r["domain"], r["samples"]))
print("\n%d scenes, 0 shared between splits" % len(rows))
for split in SPLITS:
    picked = [r for r in rows if r["split"] == split]
    ind = sum(r["samples"] for r in picked if r["domain"] == "indoor")
    out = sum(r["samples"] for r in picked if r["domain"] == "outdoor")
    print("  %-5s %2d scenes  %5d samples (indoor %4d / outdoor %4d)"
          % (split, len(picked), ind + out, ind, out))
print("  total %5d  -> written to %s" % (sum(r["samples"] for r in rows), out_csv))

# --------------------------------------------- 2. what the selection rule looks like
print("\n== per-scene scan coverage ==")
print("%-6s %-7s %-8s %7s %6s %-14s %s"
      % ("split", "scene", "domain", "images", "scans", "scan ids", "selection looks like"))
for (split, scene, dom), items in sorted(by_scene.items(),
                                         key=lambda kv: (SPLITS.index(kv[0][0]), kv[0][1])):
    scans = sorted({s for s, _, _ in items})
    contiguous = scans == list(range(scans[0], scans[0] + len(scans)))
    rng = "%d-%d" % (scans[0], scans[-1]) if len(scans) > 1 else str(scans[0])
    per_scan = Counter(s for s, _, _ in items)
    sizes = sorted(set(per_scan.values()))
    verdict = "whole scans in order" if contiguous else "scans sampled (sparse)"
    if len(sizes) > 1:
        verdict += "; RAGGED images/scan %s" % sizes
    print("%-6s %-7s %-8s %7d %6d %-14s %s"
          % (split, scene, dom, len(items), len(scans), rng, verdict))

# ------------------------------------------ 3. how depth and mask were resampled
print("\n== resize probe ==")
for split in SPLITS:
    d = os.path.join(root, split)
    ids = list_ids(d)
    step = max(1, len(ids) // PROBE)
    mask_vals, ratios, shapes, dtypes, rgb_shape = set(), [], set(), set(), None
    for sid in ids[::step][:PROBE]:
        base = os.path.join(d, sid)
        draw = np.load(base + "_depth.npy")
        mraw = np.load(base + "_depth_mask.npy")
        depth = draw.astype(np.float32).squeeze()
        mask = mraw.squeeze()
        if rgb_shape is None:
            rgb_shape = np.asarray(Image.open(base + ".jpg")).shape
        dtypes.add((str(draw.dtype), str(mraw.dtype)))
        shapes.add(depth.shape)
        mask_vals.update(np.unique(mask).tolist())
        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        r = border_ratio(depth, valid)
        if r is not None:
            ratios.append(r)
    print("  [%s] rgb %s  depth %s  dtypes %s"
          % (split, rgb_shape, sorted(shapes), sorted(dtypes)))
    print("      mask distinct values : %d -> %s" % (len(mask_vals), sorted(mask_vals)[:8]))
    if ratios:
        print("      depth border/interior: %.3f  (median over %d images)"
              % (float(np.median(ratios)), len(ratios)))

print("""
== how to read the resize probe ==
                                 nearest   bilinear
  mask distinct values                 2          4
  depth border / interior ratio    1.000      0.116

mask distinct values == 2 and border ratio near 1.000
    -> nearest neighbour was used. This is correct; nothing to redo.
mask has more than 2 values, or the border ratio is well below 1.000
    -> depth was averaged across invalid regions. That invents distances no surface
       occupies, right where the proposal describes edge bleeding, and it biases the p95
       and p99 figures. The subset would need rebuilding with nearest and the EDA re-running.""")
