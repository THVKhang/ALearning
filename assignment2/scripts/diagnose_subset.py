import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DEPTH_SUFFIX, MASK_SUFFIX, SPLITS, list_samples


def parse_id(sid):
    scene, scan, dom, h, v = sid.split("_")
    return scene, int(scan), "indoor" if dom == "indoors" else "outdoor", int(h), int(v)


def border_ratio(depth, valid):
    """Ratio of depth at border pixels (touching invalid mask) vs interior pixels.
    Nearest resize keeps ratio ~1.0; bilinear interpolation pulls ratio down (< 0.5).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    ap.add_argument("--probe", type=int, default=12, help="images per split for the resize probe")
    args = ap.parse_args()

    by_scene = defaultdict(list)
    for split in SPLITS:
        for sid in list_samples(os.path.join(args.root, split)):
            scene, scan, dom, h, v = parse_id(sid)
            by_scene[(split, scene, dom)].append((scan, h, v, sid))

    # Per-scene scan coverage
    print("== per-scene counts and scan coverage ==")
    print("%-6s %-7s %-8s %7s %7s %-18s %s"
          % ("split", "scene", "domain", "images", "scans", "scan ids", "selection looks like"))
    for key in sorted(by_scene, key=lambda k: (SPLITS.index(k[0]), k[1])):
        split, scene, dom = key
        items = sorted(by_scene[key])
        scans = sorted({s for s, _, _, _ in items})
        contiguous = scans == list(range(scans[0], scans[0] + len(scans)))
        rng = "%d-%d" % (scans[0], scans[-1]) if len(scans) > 1 else str(scans[0])
        verdict = "whole scans, in order" if contiguous else "scans sampled (sparse)"
        print("%-6s %-7s %-8s %7d %7d %-18s %s"
              % (split, scene, dom, len(items), len(scans), rng, verdict))

    # Angle grid completeness
    print("\n== angle grid completeness, per scene ==")
    for key in sorted(by_scene, key=lambda k: (SPLITS.index(k[0]), k[1])):
        split, scene, dom = key
        items = by_scene[key]
        per_scan = defaultdict(set)
        for scan, h, v, _ in items:
            per_scan[scan].add((h, v))
        sizes = sorted({len(a) for a in per_scan.values()})
        hs = sorted({h for _, h, _, _ in items})
        vs = sorted({v for _, _, v, _ in items})
        note = "uniform" if len(sizes) == 1 else "ragged (dropped individually)"
        print("  %-5s %-7s %-8s images/scan %-14s h=%d v=%d  %s"
              % (split, scene, dom, str(sizes), len(hs), len(vs), note))

    # Resize probe (depth & mask)
    print("\n== resize probe: how depth and mask were resampled ==")
    for split in SPLITS:
        d = os.path.join(args.root, split)
        ids = list_samples(d)
        step = max(1, len(ids) // args.probe)
        rgb_shape = dtypes = None
        mask_vals, ratios, shapes = set(), [], set()
        for sid in ids[::step][:args.probe]:
            base = os.path.join(d, sid)
            depth = np.load(base + DEPTH_SUFFIX).astype(np.float32).squeeze()
            mask = np.load(base + MASK_SUFFIX).squeeze()
            if rgb_shape is None:
                rgb_shape = np.asarray(Image.open(base + ".jpg")).shape
                dtypes = (np.load(base + DEPTH_SUFFIX).dtype,
                          np.load(base + MASK_SUFFIX).dtype)
            shapes.add(depth.shape)
            mask_vals.update(np.unique(mask).tolist())
            valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
            r = border_ratio(depth, valid)
            if r is not None:
                ratios.append(r)

        print("  [%s] rgb %s  depth shapes %s  dtypes depth=%s mask=%s"
              % (split, rgb_shape, sorted(shapes), dtypes[0], dtypes[1]))
        print("      mask distinct values over probe : %d -> %s"
              % (len(mask_vals), sorted(mask_vals)[:8]))
        if ratios:
            print("      depth at mask border / interior : %.3f  (median over %d images)"
                  % (float(np.median(ratios)), len(ratios)))

    print("""
== Summary ==
- Mask unique values == 2 & ratio ~ 1.0 => Nearest neighbor used (OK).
- Mask unique values > 2 or ratio < 0.5 => Bilinear/averaged resize used.""")


if __name__ == "__main__":
    main()
