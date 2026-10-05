# Read the subset selection rule back out of the subset itself.
#
# The resize and the scene partition were done outside this repository, so the rule that
# produced data/diode_subset is not written down anywhere. Section 18 of the handbook needs
# it stated, and make_subset.py needs it exactly. Rather than reconstruct it from memory,
# this script infers what it can from the files and says what remains ambiguous.
#
# Run:  python scripts/diagnose_subset.py --root data/diode_subset

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DEPTH_SUFFIX, MASK_SUFFIX, SPLITS, list_samples


def parse_id(sid):
    # 00005_00039_indoors_250_050 -> scene, scan, domain, h angle, v angle
    scene, scan, dom, h, v = sid.split("_")
    return scene, int(scan), "indoor" if dom == "indoors" else "outdoor", int(h), int(v)


def border_ratio(depth, valid):
    """Median depth of valid pixels touching an invalid pixel, over the median of interior
    valid pixels.

    DIODE stores invalid depth as 0. Resampling depth with any averaging filter mixes those
    zeros into neighbouring valid pixels and drags them toward 0; nearest neighbour copies
    single pixels and leaves them alone. Checked against a synthetic 768x1024 depth map with
    a 2 m object on an 8 m background and an invalid band, downsampled to 384x288:
    nearest gave 1.000, bilinear gave 0.116.

    Returns None when the image has no border or no interior to compare.
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

    # ------------------------------------------------ question 1 + 2: what was selected
    print("== per-scene counts and scan coverage ==")
    print("%-6s %-7s %-8s %7s %7s %-18s %s"
          % ("split", "scene", "domain", "images", "scans", "scan ids", "selection looks like"))
    for key in sorted(by_scene, key=lambda k: (SPLITS.index(k[0]), k[1])):
        split, scene, dom = key
        items = sorted(by_scene[key])
        scans = sorted({s for s, _, _, _ in items})
        contiguous = scans == list(range(scans[0], scans[0] + len(scans)))
        rng = "%d-%d" % (scans[0], scans[-1]) if len(scans) > 1 else str(scans[0])
        # A contiguous scan range starting at the scene minimum means the subset took whole
        # scans in order. A sparse range means scans were sampled.
        verdict = "whole scans, in order" if contiguous else "scans sampled (sparse)"
        print("%-6s %-7s %-8s %7d %7d %-18s %s"
              % (split, scene, dom, len(items), len(scans), rng, verdict))

    # Within a scan, is every angle combination present? If the cap was applied by dropping
    # images rather than whole scans, the angle grid will be ragged.
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
        note = "uniform" if len(sizes) == 1 else "RAGGED -> images were dropped individually"
        print("  %-5s %-7s %-8s images/scan %-14s h=%d v=%d  %s"
              % (split, scene, dom, str(sizes), len(hs), len(vs), note))

    # --------------------------------------- question 3 + 4: how depth and mask were resized
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
== how to read this ==
Both tests were calibrated on a synthetic 768x1024 depth map (a 2 m object on an 8 m
background, with an invalid band stored as 0) downsampled to 384x288:

                                    nearest   bilinear
  mask distinct values                    2          4
  depth border / interior ratio       1.000      0.116

mask distinct values is 2
    -> the mask was resampled with nearest neighbour, or thresholded afterwards. Anything
       above 2 means it was averaged, and "valid pixel" then has no clean meaning.
depth border / interior ratio near 1.0
    -> depth was resampled with nearest neighbour. This is the correct choice.
depth border / interior ratio well below 1.0
    -> depth was averaged across the invalid regions. Depth must NOT be resampled this way:
       averaging across an object boundary invents distances that no surface occupies, in
       exactly the edge-bleeding region Section 2.3 of the proposal describes, and it biases
       the p95 and p99 figures in Section 2.4. If this is what happened, the subset should be
       rebuilt with nearest and the EDA re-run before the numbers are reported again.

Not answerable from the files, so these come from whoever built the subset:
  - the interpolation used for the RGB images, which is harmless either way;
  - whether each per-scene count was a deliberate cap or simply the whole scene.""")


if __name__ == "__main__":
    main()
