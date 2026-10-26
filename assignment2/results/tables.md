## Results (val)

| Model | overall AbsRel↓ | overall RMSE↓ | overall δ1↑ | indoor AbsRel↓ | indoor RMSE↓ | indoor δ1↑ | outdoor AbsRel↓ | outdoor RMSE↓ | outdoor δ1↑ |
|---|---|---|---|---|---|---|---|---|---|
| U-Net baseline (metric) | 0.671 | 4.77 | 31.9% | 0.606 | 1.06 | 38.1% | 0.785 | 11.26 | 21.2% |
| U-Net baseline (aligned) | 0.338 | 4.43 | 58.0% | 0.165 | 0.56 | 77.4% | 0.641 | 11.21 | 24.0% |
| MiDaS_small zero-shot (aligned) | 0.321 | 4.60 | 63.2% | 0.132 | 0.49 | 83.9% | 0.651 | 11.79 | 26.9% |
| MiDaS_small full fine-tune (aligned) | 0.290 | 4.39 | 68.1% | 0.091 | 0.45 | 92.4% | 0.637 | 11.26 | 25.6% |
| MiDaS_small frozen encoder (aligned) | 0.299 | 4.34 | 66.3% | 0.106 | 0.47 | 89.6% | 0.637 | 11.11 | 25.5% |

## Compute cost

| Model | Params (M) | Trainable (M) | Epochs | Train time (min) | s / epoch | Peak train VRAM (GB) | Inference (ms/img, bs1) |
|---|---|---|---|---|---|---|---|
| U-Net baseline | 7.76 | 7.76 | 40 | 52.0 | 78 | 1.40 | 3.3 |
| MiDaS_small zero-shot | 21.32 | 0 | - | - | - | - | 9.4 |
| MiDaS_small full fine-tune | 21.32 | 21.32 | 10 | 35.6 | 214 | 2.42 | 10.4 |
| MiDaS_small frozen encoder | 21.32 | 14.90 | 10 | 30.1 | 180 | 1.17 | 11.3 |

GPU: NVIDIA GeForce RTX 4060 Laptop GPU | Python 3.14.8 | PyTorch 2.14.0+cu130 | CUDA 13.0 | cuDNN 92400 | timm 1.0.30

## Error by distance (val, aligned)

| Model | Domain | Bin | AbsRel | RMSE | δ1 | % of pixels |
|---|---|---|---|---|---|---|
| U-Net baseline | indoor | near (<5m) | 0.154 | 0.82 | 79.2% | 97.8% |
| U-Net baseline | indoor | mid (5-20m) | 0.563 | 7.88 | 22.7% | 2.2% |
| U-Net baseline | indoor | far (>20m) | 0.908 | 30.76 | 0.1% | 0.0% |
| U-Net baseline | outdoor | near (<5m) | 0.947 | 2.52 | 38.2% | 30.2% |
| U-Net baseline | outdoor | mid (5-20m) | 0.357 | 5.58 | 29.8% | 54.7% |
| U-Net baseline | outdoor | far (>20m) | 0.723 | 27.23 | 0.4% | 15.1% |
| MiDaS_small zero-shot | indoor | near (<5m) | 0.119 | 0.52 | 85.1% | 97.8% |
| MiDaS_small zero-shot | indoor | mid (5-20m) | 0.598 | 10.27 | 50.2% | 2.2% |
| MiDaS_small zero-shot | indoor | far (>20m) | 0.898 | 30.68 | 0.8% | 0.0% |
| MiDaS_small zero-shot | outdoor | near (<5m) | 0.958 | 3.98 | 44.5% | 30.2% |
| MiDaS_small zero-shot | outdoor | mid (5-20m) | 0.357 | 7.94 | 33.4% | 54.7% |
| MiDaS_small zero-shot | outdoor | far (>20m) | 0.732 | 28.29 | 0.4% | 15.1% |
| MiDaS_small full fine-tune | indoor | near (<5m) | 0.077 | 0.57 | 93.6% | 97.8% |
| MiDaS_small full fine-tune | indoor | mid (5-20m) | 0.636 | 11.51 | 58.2% | 2.2% |
| MiDaS_small full fine-tune | indoor | far (>20m) | 0.896 | 30.61 | 1.0% | 0.0% |
| MiDaS_small full fine-tune | outdoor | near (<5m) | 0.942 | 3.45 | 43.5% | 30.2% |
| MiDaS_small full fine-tune | outdoor | mid (5-20m) | 0.349 | 6.82 | 31.8% | 54.7% |
| MiDaS_small full fine-tune | outdoor | far (>20m) | 0.716 | 28.10 | 0.5% | 15.1% |
| MiDaS_small frozen encoder | indoor | near (<5m) | 0.092 | 0.49 | 91.1% | 97.8% |
| MiDaS_small frozen encoder | indoor | mid (5-20m) | 0.600 | 10.74 | 57.0% | 2.2% |
| MiDaS_small frozen encoder | indoor | far (>20m) | 0.887 | 30.32 | 1.0% | 0.0% |
| MiDaS_small frozen encoder | outdoor | near (<5m) | 0.949 | 3.18 | 44.4% | 30.2% |
| MiDaS_small frozen encoder | outdoor | mid (5-20m) | 0.347 | 6.07 | 31.1% | 54.7% |
| MiDaS_small frozen encoder | outdoor | far (>20m) | 0.708 | 27.34 | 0.3% | 15.1% |
