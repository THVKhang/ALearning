# Assignment 2 – Monocular Depth Estimation (DIODE)

Group CR7. Full proposal and results: [assignment page](index.html).

## Folder layout
```
configs/          base.yaml (shared seed / batch / lr / paths) + one yaml per experiment
src/data/         dataset + dataloaders (masks, 0.6–50 m indoor / 0.6–200 m outdoor caps)
src/models/       U-Net baseline, MiDaS_small wrapper (optional frozen encoder)
src/engine/       losses, metrics, scale/shift alignment, trainer
src/utils/        seed, config loading, env info
train.py          train one experiment
evaluate.py       evaluate a checkpoint (or MiDaS zero-shot) on val / test
scripts/data/     subset building, EDA, dataset checks (M1)
scripts/report/   result tables, loss curves, error analysis, qualitative figure
tests/            metric / alignment / model / pipeline checks
results/          tables and figures used on the assignment page
```

## Setup
```bash
pip install -r requirements.txt
```
Tested with Python 3.14.8, PyTorch 2.14.0 + CUDA 13.0, timm 1.0.30, RTX 4060 Laptop (8 GB).
MiDaS_small is downloaded through `torch.hub` (intel-isl/MiDaS) on first use.

## Data
DIODE subset (7,195 RGB–depth pairs, 384×288, split by scene). Not in git. Build it with
`scripts/data/make_subset.py` from the DIODE release, or put the prepared subset here:
```
data/diode_subset/{train,val,test}/<id>.jpg, <id>_depth.npy, <id>_depth_mask.npy
```

## Run (from `assignment2/` or repo root)
```bash
python tests/test_metrics.py                               # metric + alignment sanity checks
python tests/test_models.py                                # U-Net + MiDaS_small shape/grad checks
python tests/test_pipeline.py                              # synthetic end-to-end Trainer check

python evaluate.py --config configs/midas_zeroshot.yaml    # MiDaS zero-shot on val
python train.py    --config configs/baseline_unet.yaml     # U-Net from scratch (~1 h budget)
python train.py    --config configs/midas_full.yaml        # MiDaS full fine-tune
python train.py    --config configs/midas_frozen.yaml      # MiDaS frozen encoder
python evaluate.py --config configs/baseline_unet.yaml     # uses outputs/<name>/best.pth
python evaluate.py --config configs/midas_full.yaml --split test
python evaluate.py --config configs/midas_frozen.yaml --split test

# Multi-seed controlled experiment (M3 Final: seeds 42, 43, 44)
python train.py    --config configs/midas_full.yaml --seed 43
python train.py    --config configs/midas_frozen.yaml --seed 43

# Diagnostic & report scripts
python scripts/report/erode_check.py                       # boundary-mask erosion (0, 1, 2 px)
python scripts/report/outdoor_check.py                     # per-scene outdoor correlation check
python scripts/report/edge_and_alignment_check.py          # edge vs. flat + median vs. LS alignment
python scripts/report/make_report.py                       # results/tables.md, loss curves, error plot
python scripts/report/qualitative.py                       # results/qualitative.png
```
Or everything at once: `bash run_all.sh`. Add `--smoke` to `train.py` for a 1-minute pipeline check.

Each run writes to `outputs/<name>/`: `config.yaml`, `history.csv` (loss, val metrics, epoch
time, VRAM per epoch), `best.pth`, `train_summary.json`, `eval_val.json`.

## Reproducibility
- Seed 42 for python / numpy / torch / dataloader workers, `cudnn.deterministic=True`,
  `benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
  Two smoke runs of the same config gave identical loss and metrics.
- Checkpoint rule: keep the epoch with the lowest overall val Abs Rel.
- MiDaS outputs relative inverse depth, so before scoring it gets a per-image least-squares
  scale and shift in disparity space against 1/GT on valid pixels. The U-Net is reported both
  raw (metres) and with the same alignment.
- Checkpoints are not in git (too large); rerun the commands above to recreate them.

## Draft milestone
Commit for M2 Draft is tagged `a2-m2-draft`.
