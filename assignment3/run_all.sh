#!/usr/bin/env bash
# End-to-end Assignment 3 pipeline: tests -> EDA -> 4 mandatory models + 2 ablations -> evaluation -> report figures
set -e

python tests/test_multimodal.py
python scripts/eda.py

for cfg in rgb_only depth_only simple_fusion improved_fusion ablation_no_gate ablation_no_aux; do
  python train.py --config "configs/${cfg}.yaml"
  python evaluate.py --config "configs/${cfg}.yaml" --split val
  python evaluate.py --config "configs/${cfg}.yaml" --split test
done

python scripts/make_report.py
