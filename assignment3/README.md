# Assignment 3 – Multimodal Deep Learning: Representation, Fusion, and Evaluation

Group CR7 (`Trần Hoàng Vỹ Khang – 2352502`, `Đỗ Đăng Khoa – 2352554`, `Dương Đăng Khoa – 2352551`).  
Full assignment page: [index.html](index.html) | Full report: [CR7_A3_Report.md](CR7_A3_Report.md) | AI disclosure: [AI_USAGE.md](AI_USAGE.md).

---

## 1. Overview & Task Formulation

- **Task:** 6-Class Multimodal Environment Domain & Camera-Elevation Regime Classification from co-registered **RGB Appearance (Modality A)** and **3D LiDAR Depth + Validity Geometry (Modality B)** on the **DIODE** dataset (`7,195` aligned pairs across `15` scenes and `144` scans).
- **Six Target Classes (`{Indoor, Outdoor} × {Level 000–010°, Mid-Elevated 020–030°, High-Elevated 040–050°}`):**
  - `0`: `Indoor-Level (00-10°)` (`1,334` pairs)
  - `1`: `Indoor-Mid (20-30°)` (`1,417` pairs)
  - `2`: `Indoor-High (40-50°)` (`1,255` pairs)
  - `3`: `Outdoor-Level (00-10°)` (`1,051` pairs)
  - `4`: `Outdoor-Mid (20-30°)` (`1,146` pairs)
  - `5`: `Outdoor-High (40-50°)` (`992` pairs)
- **Why Multimodal?**
  - **Modality A (`RGB`, `3 × 72 × 96`)** captures semantic scene appearance (foliage, brick, sky vs. indoor walls, furniture, lighting), achieving **90.91% Domain Accuracy** on unseen test scenes, but struggles to infer camera elevation angle on textureless walls or uniform tree canopies (**70.36% Elevation Accuracy**).
  - **Modality B (`Depth-Geometry`, `3 × 72 × 96`: normalized log-depth $\tilde{d}$, binary LiDAR validity mask $m$, and vertical surface slant $\partial_y \tilde{d}$)** directly measures 3D range and floor/ceiling surface tilt (**95.45% Elevation Accuracy** on clean indoor rooms), but suffers when outdoor upward angles lose **37.8%–37.9%** of laser returns to open sky (`Outdoor-High F1 = 73.63%`).

---

## 2. Project Layout

```text
assignment3/
├── configs/                    base.yaml + 4 mandatory model configs + 2 controlled ablations
├── proposal/scenes_used.csv    15-scene manifest (7,195 paired samples, 0% scene overlap)
├── src/
│   ├── data/dataset.py         paired RGB + 3-channel LiDAR geometry dataset & caching
│   ├── models/encoders.py      symmetric 4-stage spatial ConvNet encoders
│   ├── fusion/modules.py       late concat, bidirectional cross-attention, validity-conditioned gate
│   ├── models/multimodal.py    Unimodal, SimpleFusion, and ImprovedMultimodal classifiers
│   ├── engine/{loss,trainer}.py joint multi-task loss, AMP trainer, checkpointing by val macro-F1
│   └── utils/{common,metrics}.py deterministic seeding, config loader, multi-task metrics
├── scripts/
│   ├── eda.py                  pair-validity check, scene manifest, multimodal EDA plots
│   └── make_report.py          comparison tables, support/conflict taxonomy, curves & figures
├── tests/test_multimodal.py    unit tests for all 4 model families, gradients, and metrics
├── assets/                     publication figures used on index.html and CR7_A3_Report.md
├── results/tables.md           generated markdown tables from outputs/ JSON artifacts
├── outputs/                    tracked JSON/CSV run artifacts (weights git-ignored)
├── train.py                    CLI training entry point
├── evaluate.py                 CLI evaluation + missing-modality stress test entry point
└── run_all.sh                  reproduce all tests, EDA, 6 runs, evaluations, and figures
```

---

## 3. Setup & Reproducibility Commands

```bash
pip install -r assignment3/requirements.txt
```

Run from `assignment3/` (or execute `bash run_all.sh`):

```bash
# 1. Unit tests & Multimodal EDA (verifies 7,195 pairs & builds fast 72x96 tensor cache)
python tests/test_multimodal.py
python scripts/eda.py

# 2. Train & evaluate the 4 mandatory models + 2 controlled ablations (seed 42)
for cfg in rgb_only depth_only simple_fusion improved_fusion ablation_no_gate ablation_no_aux; do
  python train.py --config "configs/${cfg}.yaml"
  python evaluate.py --config "configs/${cfg}.yaml" --split val
  python evaluate.py --config "configs/${cfg}.yaml" --split test
done

# 3. Generate all comparison tables, support/conflict breakdown, and figures
python scripts/make_report.py
```

---

## 4. Main Experimental Results (`seed = 42`)

### 4.1 Quantitative Comparison (Val & Held-Out Test Splits)

| Model | Params | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Domain Acc | Test Elev Acc | Train Time (s) | Infer (ms/pair) |
|---|---|---|---|---|---|---|---|---|---|
| Unimodal A: RGB-Only ConvNet | 463,974 | 68.18% | 69.38% | 64.73% | 65.62% | 90.91% | 70.36% | 62.0s (5.17s/ep) | 1.787 |
| Unimodal B: Depth-Geometry ConvNet | 463,974 | 83.73% | 80.20% | **92.91%** | **90.84%** | 97.45% | **95.45%** | 63.4s (5.28s/ep) | 1.796 |
| Simple Fusion: Late Concatenation | 960,326 | 70.55% | 71.14% | 68.73% | 71.30% | 95.55% | 71.91% | 111.9s (9.31s/ep) | 4.841 |
| Improved Fusion: Cross-Attn + Validity Gate | 1,137,236 | 81.55% | 80.87% | 87.91% | 87.12% | 96.36% | 91.18% | 99.1s (8.25s/ep) | 5.548 |
| Ablation 1: No Cross-Attn / No Validity Gate | 978,258 | **85.91%** | **83.91%** | 92.00% | 90.64% | **97.73%** | 94.09% | 112.2s (9.33s/ep) | 5.117 |
| Ablation 2: No Aux Unimodal Loss / No Modality Drop | 1,137,236 | 73.27% | 73.76% | 73.55% | 74.74% | 95.64% | 76.36% | 122.6s (10.21s/ep) | 7.409 |

### 4.2 Per-Class F1 on Held-Out Test Split (`n = 1,100`)

| Class | Unimodal A (RGB) | Unimodal B (Depth) | Simple Fusion | Improved Fusion |
|---|---|---|---|---|
| Indoor-Level (00-10°) | 67.09% | **97.22%** | 65.08% | 88.36% |
| Indoor-Mid (20-30°) | 51.27% | **96.68%** | 53.24% | 86.52% |
| Indoor-High (40-50°) | 60.53% | **97.12%** | 63.62% | 91.98% |
| Outdoor-Level (00-10°) | 83.92% | **96.45%** | 91.03% | 90.17% |
| Outdoor-Mid (20-30°) | 71.71% | 83.91% | 80.39% | **85.27%** |
| Outdoor-High (40-50°) | 59.20% | 73.63% | 74.47% | **80.41%** |

### 4.3 Cross-Modal Support vs. Conflict Breakdown on Test Split (`n = 1,100`)

| Unimodal Agreement Regime (Test Split, n=1,100) | Test Samples (%) | Simple Fusion Correct | Improved Fusion Correct |
|---|---|---|---|
| Both Unimodal Correct (RGB ✓, Depth ✓) | 671 (61.0%) | 578/671 (86.1%) | **643/671 (95.8%)** |
| Conflict: RGB Correct, Depth Wrong (RGB ✓, Depth ✗) | 41 (3.7%) | 27/41 (65.9%) | **30/41 (73.2%)** |
| Conflict: Depth Correct, RGB Wrong (RGB ✗, Depth ✓) | 351 (31.9%) | 140/351 (39.9%) | **275/351 (78.3%)** |
| Both Unimodal Wrong — Emergent Support (RGB ✗, Depth ✗) | 37 (3.4%) | 11/37 (29.7%) | **19/37 (51.4%)** |

### 4.4 Missing-Modality Stress Test & LiDAR Validity Regime Breakdown

| Fusion Model | Full Test Macro-F1 | Clean LiDAR (≥65% valid, n=900) | Noisy LiDAR (<65% valid, n=200) | Drop Depth at Test (0% LiDAR) | Drop RGB at Test (0% RGB) |
|---|---|---|---|---|---|
| Simple Fusion: Late Concatenation | 71.30% | 65.37% | 67.10% | 3.41% | 44.77% |
| Improved Fusion: Cross-Attn + Validity Gate | 87.12% | 84.74% | **91.92%** | 59.36% | 73.24% |
| Ablation 1: No Cross-Attn / No Validity Gate | **90.64%** | **87.99%** | 89.66% | **63.99%** | **76.76%** |
| Ablation 2: No Aux Unimodal Loss / No Modality Drop | 74.74% | 69.89% | 91.09% | 23.59% | 49.95% |

---

## 5. Hardware & Software Environment

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Software:** Python `3.13.9`, PyTorch `2.14.0+cu130`, CUDA `13.0`, cuDNN `92400`, NumPy `2.5.3`
- **Reproducibility:** `seed = 42`, `cudnn.deterministic = True`, `cudnn.benchmark = False`. All JSON/CSV run artifacts are tracked under `outputs/<run_name>/`.
