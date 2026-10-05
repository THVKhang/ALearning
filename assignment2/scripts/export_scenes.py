# Rebuild proposal/scenes_used.csv from the data and check the splits are scene-disjoint.
# The committed CSV is the evidence for the no-leakage claim, so it has to be derivable
# from the data rather than typed out by hand.

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import SPLITS, list_samples


def scene_of(sample_id):
    # e.g. 00005_00039_indoors_250_050 -> ("00005", "indoor")
    scene, _, dom = sample_id.split("_")[:3]
    return scene, "indoor" if dom == "indoors" else "outdoor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/diode_subset")
    ap.add_argument("--out", default="proposal/scenes_used.csv")
    args = ap.parse_args()

    counts = Counter()
    for split in SPLITS:
        for sid in list_samples(os.path.join(args.root, split)):
            scene, domain = scene_of(sid)
            counts[(scene, split, domain)] += 1

    rows = sorted(({"scene": s, "split": sp, "domain": d, "samples": n}
                   for (s, sp, d), n in counts.items()), key=lambda r: r["scene"])

    # A scene in two splits is exactly the leakage the scene-level split exists to stop.
    seen = {}
    for r in rows:
        if r["scene"] in seen:
            raise SystemExit("LEAKAGE: scene %s is in both %s and %s"
                             % (r["scene"], seen[r["scene"]], r["split"]))
        seen[r["scene"]] = r["split"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("scene", "split", "domain", "samples"))
        w.writeheader()
        w.writerows(rows)

    print("%d scenes, 0 shared between splits" % len(rows))
    for split in SPLITS:
        picked = [r for r in rows if r["split"] == split]
        print("  %-5s %2d scenes  %5d samples"
              % (split, len(picked), sum(r["samples"] for r in picked)))
    print("  total %5d samples -> %s"
          % (sum(r["samples"] for r in rows), args.out))


if __name__ == "__main__":
    main()
