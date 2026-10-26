# Multimodal EDA Summary – Assignment 3 (DIODE Paired RGB + 3D LiDAR Depth)

## 1. Cross-Modal Pair Validity Check (Handbook Section 28.1)
- **Total paired samples verified:** 7,195 (`train`: 4,995 across 11 scenes; `val`: 1,100 across 2 scenes; `test`: 1,100 across 2 scenes).
- **Missing or unaligned files:** 0 (100.0% of samples contain co-registered RGB `.jpg`, LiDAR `_depth.npy`, and `_depth_mask.npy` captured from the same FARO Focus S350 scanner pose).
- **Scene-level split leakage check:** 0 scenes shared between `train`, `val`, and `test`.

## 2. Six-Class Distribution & LiDAR Validity Rate by Split
| Class ID | Class Name | Train (n=4,995) | Val (n=1,100) | Test (n=1,100) | Total (n=7,195) | Train Mean Valid LiDAR % | Train Mean Depth (m) |
|---|---|---|---|---|---|---|---|
| 0 | Indoor-Level (00-10°) | 934 (18.7%) | 183 (16.6%) | 217 (19.7%) | 1334 (18.5%) | 98.1% | 4.00 m |
| 1 | Indoor-Mid (20-30°) | 908 (18.2%) | 274 (24.9%) | 235 (21.4%) | 1417 (19.7%) | 98.7% | 3.66 m |
| 2 | Indoor-High (40-50°) | 764 (15.3%) | 243 (22.1%) | 248 (22.5%) | 1255 (17.4%) | 99.4% | 2.75 m |
| 3 | Outdoor-Level (00-10°) | 775 (15.5%) | 134 (12.2%) | 142 (12.9%) | 1051 (14.6%) | 84.8% | 11.53 m |
| 4 | Outdoor-Mid (20-30°) | 851 (17.0%) | 140 (12.7%) | 155 (14.1%) | 1146 (15.9%) | 62.2% | 14.50 m |
| 5 | Outdoor-High (40-50°) | 763 (15.3%) | 126 (11.5%) | 103 (9.4%) | 992 (13.8%) | 62.1% | 9.19 m |
