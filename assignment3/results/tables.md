## 1. Main Quantitative Comparison (Val & Held-Out Test Splits)

| Model | Params | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Domain Acc | Test Elev Acc | Train Time (s) | Infer (ms/pair) |
|---|---|---|---|---|---|---|---|---|---|
| Unimodal A: RGB-Only ConvNet | 463,974 | 68.18% | 69.38% | 64.73% | 65.62% | 90.91% | 70.36% | 62.0s (5.17s/ep) | 1.787 |
| Unimodal B: Depth-Geometry ConvNet | 463,974 | 83.73% | 80.20% | 92.91% | 90.84% | 97.45% | 95.45% | 63.4s (5.28s/ep) | 1.796 |
| Simple Fusion: Late Concatenation | 960,326 | 70.55% | 71.14% | 68.73% | 71.30% | 95.55% | 71.91% | 111.9s (9.31s/ep) | 4.841 |
| Improved Fusion: Cross-Attn + Validity Gate | 1,137,236 | 81.55% | 80.87% | 87.91% | 87.12% | 96.36% | 91.18% | 99.1s (8.25s/ep) | 5.548 |
| Ablation 1: No Cross-Attn / No Validity Gate | 978,258 | 85.91% | 83.91% | 92.00% | 90.64% | 97.73% | 94.09% | 112.2s (9.33s/ep) | 5.117 |
| Ablation 2: No Aux Unimodal Loss / No Modality Drop | 1,137,236 | 73.27% | 73.76% | 73.55% | 74.74% | 95.64% | 76.36% | 122.6s (10.21s/ep) | 7.409 |

## 2. Per-Class F1 on Held-Out Test Split

| Class | Unimodal A | Unimodal B | Simple Fusion | Improved Fusion |
|---|---|---|---|---|
| Indoor-Level (00-10°) | 67.09% | **97.22%** | 65.08% | 88.36% |
| Indoor-Mid (20-30°) | 51.27% | **96.68%** | 53.24% | 86.52% |
| Indoor-High (40-50°) | 60.53% | **97.12%** | 63.62% | 91.98% |
| Outdoor-Level (00-10°) | 83.92% | **96.45%** | 91.03% | 90.17% |
| Outdoor-Mid (20-30°) | 71.71% | 83.91% | 80.39% | **85.27%** |
| Outdoor-High (40-50°) | 59.20% | 73.63% | 74.47% | **80.41%** |

## 3. Cross-Modal Support vs. Conflict Breakdown on Held-Out Test Split

| Unimodal Agreement Regime (Test Split, n=1,100) | Test Samples (%) | Simple Fusion Correct | Improved Fusion Correct |
|---|---|---|---|
| Both Unimodal Correct (RGB ✓, Depth ✓) | 671 (61.0%) | 578/671 (86.1%) | 643/671 (95.8%) |
| Conflict: RGB Correct, Depth Wrong (RGB ✓, Depth ✗) | 41 (3.7%) | 27/41 (65.9%) | 30/41 (73.2%) |
| Conflict: Depth Correct, RGB Wrong (RGB ✗, Depth ✓) | 351 (31.9%) | 140/351 (39.9%) | 275/351 (78.3%) |
| Both Unimodal Wrong — Emergent Support (RGB ✗, Depth ✗) | 37 (3.4%) | 11/37 (29.7%) | 19/37 (51.4%) |

## 4. Missing-Modality Stress Test & LiDAR Validity Regime Breakdown

| Fusion Model | Full Test Macro-F1 | Clean LiDAR (≥65% valid) | Noisy LiDAR (<65% valid) | Drop Depth at Test (0% LiDAR) | Drop RGB at Test (0% RGB) |
|---|---|---|---|---|---|
| Simple Fusion: Late Concatenation | 71.30% | 65.37% | 67.10% | 3.41% | 44.77% |
| Improved Fusion: Cross-Attn + Validity Gate | 87.12% | 84.74% | 91.92% | 59.36% | 73.24% |
| Ablation 1: No Cross-Attn / No Validity Gate | 90.64% | 87.99% | 89.66% | 63.99% | 76.76% |
| Ablation 2: No Aux Unimodal Loss / No Modality Drop | 74.74% | 69.89% | 91.09% | 23.59% | 49.95% |
