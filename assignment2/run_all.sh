#!/usr/bin/env bash
# Run every M2/M3 experiment in order, then build tables and figures. Run from assignment2/.
set -e

python tests/test_metrics.py
python tests/test_models.py
python tests/test_pipeline.py

python evaluate.py --config configs/midas_zeroshot.yaml --split val
python evaluate.py --config configs/midas_zeroshot.yaml --split test

for seed in 42 43 44; do
  python train.py --config configs/baseline_unet.yaml --seed "$seed"
  python evaluate.py --config configs/baseline_unet.yaml --seed "$seed" --split val
  python evaluate.py --config configs/baseline_unet.yaml --seed "$seed" --split test

  python train.py --config configs/midas_full.yaml --seed "$seed"
  python evaluate.py --config configs/midas_full.yaml --seed "$seed" --split val
  python evaluate.py --config configs/midas_full.yaml --seed "$seed" --split test

  python train.py --config configs/midas_frozen.yaml --seed "$seed"
  python evaluate.py --config configs/midas_frozen.yaml --seed "$seed" --split val
  python evaluate.py --config configs/midas_frozen.yaml --seed "$seed" --split test
done

python scripts/report/erode_check.py
python scripts/report/outdoor_check.py
python scripts/report/edge_and_alignment_check.py --split val
python scripts/report/edge_and_alignment_check.py --split test
python scripts/report/make_report.py
python scripts/report/qualitative.py

