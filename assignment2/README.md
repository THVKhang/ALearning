# Assignment 2 – Monocular Depth Estimation (DIODE)

Group CR7. Full proposal and results: [assignment page](index.html).

## Data
DIODE subset (7,195 RGB–depth pairs, 384×288, split by scene). Not in git. Put it here:
```
data/diode_subset/{train,val,test}/<id>.jpg, <id>_depth.npy, <id>_depth_mask.npy
```

## Scripts (run from `assignment2/`)
```bash
python scripts/check_dataset.py     --root data/diode_subset   # shapes / depth range / mask check
python scripts/visualize_samples.py --root data/diode_subset   # RGB | depth | mask figure
python scripts/eda.py               --root data/diode_subset   # EDA tables + figures -> outputs/eda/
python scripts/benchmark_midas.py   --root data/diode_subset --train_iters 100   # speed / VRAM test
```

Needs: `torch`, `numpy`, `pillow`, `matplotlib`, `timm` (for MiDaS from `torch.hub`).
