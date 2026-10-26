#!/usr/bin/env bash
# Run every M2 experiment in order, then build tables and figures. Run from assignment2/.
set -e

python tests/test_metrics.py

python evaluate.py --config configs/midas_zeroshot.yaml

python train.py --config configs/baseline_unet.yaml
python evaluate.py --config configs/baseline_unet.yaml

python train.py --config configs/midas_full.yaml
python evaluate.py --config configs/midas_full.yaml

python train.py --config configs/midas_frozen.yaml
python evaluate.py --config configs/midas_frozen.yaml

python scripts/report/make_report.py
python scripts/report/qualitative.py
