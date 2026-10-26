# AI Usage Disclosure – Assignment 2

**Course:** CO3133 – Deep Learning and Its Applications  
**Semester:** 261  
**Group:** CR7 (Trần Hoàng Vỹ Khang, Đỗ Đăng Khoa, Dương Đăng Khoa)

---

## General Policy Declaration

All core problem formulations, data sampling strategies, pipeline designs, model implementations, experimental methodologies, and report conclusions are written, directed and owned by group CR7. The dataset choice, the scene-level split, the U-Net and MiDaS implementations, the controlled-experiment design and every number reported on the assignment page come from code written and executed by the group on group hardware.

Across all stages of Assignment 2, AI tools acted strictly as a **code reviewer and architecture controller**, not as a direct code modifier or autonomous author: reviewing member-written files for cleanliness and edge cases, controlling model and pipeline architecture (valid-pixel masking, log-depth output activation, encoder + BatchNorm freezing, per-image scale/shift alignment), diagnosing resampling and sensor-range edge cases, and verifying compliance with the course handbook. Every edit, script execution, and verification step was performed directly by the responsible group member, as handbook Section 5.5 requires.

---

## M1 Proposal Disclosure Summary

| # | Task / Stage | Purpose / Category | Affected Files | Verification Method | Member Responsible |
|---|---|---|---|---|---|
| **1** | M1 Data Exploration | Code review of data validation & depth mask checks | `src/data/dataset.py`, `scripts/data/check_dataset.py` | Local Python script verification & PyTorch docs | Đỗ Đăng Khoa |
| **2** | M1 Hardware Benchmark | Debugging review for Torch Hub non-interactive issue | `scripts/benchmark_midas.py` | Local execution on RTX 4060 GPU & terminal log | Đỗ Đăng Khoa |
| **3** | M1 Visualization | Review of colormap scaling & 99th percentile capping | `scripts/data/visualize_samples.py` | Visual inspection of `assets/train_samples.png` | Đỗ Đăng Khoa |
| **4** | M1 Statistical EDA | Review of memory optimization for quantile computation | `scripts/data/eda.py` | Execution runtime check & CSV log verification | Đỗ Đăng Khoa |
| **5** | M1 Experiment Setup | Architecture & controlled-experiment seed isolation review | `assignment2/index.html` Sec. 4 | Handbook Sec. 18 & PyTorch seed docs | Đỗ Đăng Khoa |
| **6** | M1 Leakage Check | Review of scene-level data split disjunction verification | `proposal/scenes_used.csv`, `scripts/data/export_scenes.py`, Report Sec. 2.2 | Set intersection check in Python & CSV lookup | Đỗ Đăng Khoa |
| **7** | M1 Loss Formulation | Mathematical review of SILog loss formula | Report Sec. 3.2 & Sec. 5 | Eigen et al. (2014) NIPS paper cross-reference | Đỗ Đăng Khoa |
| **8** | M1 Proposal Report | Review of report structure & 16-point handbook checklist | `assignment2/index.html` | Visual check at `http://localhost:8085` & Handbook Sec. 18 | Đỗ Đăng Khoa |
| **9** | M1 Compliance Audit | Handbook Section 18 & 21 compliance review (license, scene manifest, decision criterion) | `scripts/data/export_scenes.py`, `proposal/scenes_used.csv`, `assignment2/index.html` | Handbook Sec. 18, 19.6, 20, 21, 22; diode-dataset.org | Trần Hoàng Vỹ Khang |
| **10** | M1 Subset Reproducibility | Review of subset selection rule & diagnosis of bilinear depth resampling | `scripts/data/make_subset.py`, `scripts/data/kaggle_diagnose.py`, `notebooks/`, `assignment2/index.html` | Group Kaggle notebook; OpenCV docs; regenerated EDA | Trần Hoàng Vỹ Khang |

---

## M2 Draft Disclosure Summary

| # | Task / Stage | Purpose / Category | Affected Files | Verification Method | Member Responsible |
|---|---|---|---|---|---|
| **11** | M2 Loss & Metrics | Code & mathematical review of loss functions and edge-case tests | `src/engine/loss.py`, `src/engine/metrics.py`, `tests/test_metrics.py` | Unit tests; Eigen et al. (2014) | Dương Đăng Khoa |
| **12** | M2 Baseline U-Net | Architecture control & review of U-Net and training pipeline | `src/models/unet.py`, `src/engine/trainer.py`, `train.py`, `tests/test_models.py`, `tests/test_pipeline.py` | Synthetic pipeline test; Ronneberger et al. (2015) | Dương Đăng Khoa |
| **13** | M2 Architecture Control & Evaluation Review | Architecture review of MiDaS fine-tuning, scale/shift alignment, metric clamping, and error-analysis diagnostics | `src/engine/{metrics,align,loss,trainer}.py`, `src/models/{unet,midas,__init__}.py`, `src/utils/common.py`, `src/data/dataset.py`, `configs/*.yaml`, `train.py`, `evaluate.py`, `scripts/report/*.py` | Unit tests; deterministic repeat-run check; full runs on RTX 4060 | Đỗ Đăng Khoa |

---

## M1 Detailed Log Entries

### Log Entry 1: Data Pipeline Validation & Mask Format Spot-Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Gemini / Claude model) |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Data Exploration (04 October 2026) |
| **Purpose / Category** | Data validation & coding assistance (checking DIODE depth mask array types and depth caps) |
| **Affected Sections / Files** | `assignment2/src/data/dataset.py`, `assignment2/scripts/data/check_dataset.py`, Report Sec. 2 |
| **Representative Prompt** | *"Kiểm tra giúp mình xem format file depth_mask.npy của DIODE là 0/1 hay boolean, và xem cách đọc float16 depth có bị trôi precision không."* |
| **AI Contribution** | Suggested converting mask arrays explicitly to uint8 `(mask > 0)` and applying depth threshold caps (50m indoor / 200m outdoor) to avoid invalid zero-division in loss calculation. |
| **Student Verification & Editing** | Tested array loading on 100 sample images via `scripts/data/check_dataset.py`, verified depth min/max values, and confirmed mask values strictly equal 0 or 1. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | DIODE official devkit documentation, PyTorch Dataset API docs, local script `scripts/data/check_dataset.py`. |

---

### Log Entry 2: Debugging Torch Hub Non-Interactive Execution

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Hardware Benchmarking (04 October 2026) |
| **Purpose / Category** | Debugging & performance measurement |
| **Affected Sections / Files** | `assignment2/scripts/benchmark_midas.py`, Report Sec. 4 & Sec. 7 |
| **Representative Prompt** | *"Script benchmark MiDaS bị treo khi gọi torch.hub.load('intel-isl/MiDaS', 'MiDaS_small') trong môi trường non-interactive, cách fix như thế nào?"* |
| **AI Contribution** | Identified that `torch.hub.load` was prompting for interactive confirmation when downloading dependencies (EfficientNet backbone repo `rwightman/gen-efficientnet-pytorch`) and recommended passing `trust_repo=True`. |
| **Student Verification & Editing** | Updated `benchmark_midas.py` with `trust_repo=True`, ran full 100-iter benchmark on local NVIDIA RTX 4060 GPU, and verified VRAM and latency output logs. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | PyTorch `torch.hub` official documentation & local GPU execution terminal log. |

---

### Log Entry 3: Depth Map Visualization & Percentile Scaling

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Visual Inspection (04 October 2026) |
| **Purpose / Category** | Figure generation & visualization formatting |
| **Affected Sections / Files** | `assignment2/scripts/data/visualize_samples.py`, Report Sec. 3.1 & Figure 1 |
| **Representative Prompt** | *"Khi vẽ depth map bằng matplotlib, dùng `vmax=max_depth` bị lỗi 1 pixel nhiễu kéo cả ảnh về 1 màu tối. Có cách chọn colormap và scaling nào đẹp và đúng bản chất dữ liệu không?"* |
| **AI Contribution** | Recommended setting upper colormap bound `vmax` to each image's 99th percentile depth (`np.percentile(valid_depths, 99)`) and setting invalid mask pixels to black for crisp object boundary contrast. |
| **Student Verification & Editing** | Implemented percentile scaling in `scripts/data/visualize_samples.py`, generated `assets/train_samples.png`, and visually inspected image alignment across all 5 random samples. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | Matplotlib colormap API documentation & visual output image verification. |

---

### Log Entry 4: EDA Statistical Computation & RAM Optimization

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal EDA Analysis (04 October 2026) |
| **Purpose / Category** | Result analysis & coding optimization |
| **Affected Sections / Files** | `assignment2/scripts/data/eda.py`, Report Sec. 3.2 & Figures 2–4 |
| **Representative Prompt** | *"Hàm tính percentile trên 7.195 mẫu depthmap của DIODE bị chậm và ngốn RAM khi dồn hết numpy array vào RAM. Có cách nào tính percentile theo batch rồi tổng hợp lại không?"* |
| **AI Contribution** | Suggested chunking EDA calculations scene-by-scene, accumulating per-image summary statistics (p5, median, p95, p99, invalid %) into lightweight dictionary records, and exporting to CSV. |
| **Student Verification & Editing** | Executed `scripts/data/eda.py` locally, confirmed execution time under 45 seconds, verified generated plots (`depth_hist.png`, `invalid_hist.png`, `angle_by_scene.png`), and checked CSV log for 7,195 valid rows. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | NumPy array memory optimization guides & generated dataset CSV metadata. |

---

### Log Entry 5: Controlled Experiment Setup & Seed Isolation Review

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Experiment Design (04 October 2026) |
| **Purpose / Category** | Architecture & experiment design review |
| **Affected Sections / Files** | `assignment2/index.html` Sec. 4 & Sec. 6 |
| **Representative Prompt** | *"Rà soát giúp mình phần Controlled Experiment: Option A (Frozen encoder) vs Option B (Full fine-tuning) xem cách chia random seed và giữ nguyên hyperparameters đã đủ chặt chẽ chưa."* |
| **AI Contribution** | Reviewed the controlled matrix, confirming that using 3 fixed random seeds (42, 43, 44), identical SILog loss, and identical batch size (8) isolates trainable parameter status as the single variable factor. |
| **Student Verification & Editing** | Cross-referenced matrix against handbook Section 18 controlled experiment criteria and finalized experiment design table in report. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | CO3133 Course Handbook Section 18 & PyTorch reproducibility best practices. |

---

### Log Entry 6: Scene-Level Data Split & Leakage Disjunction Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Data Splitting (04 October 2026) |
| **Purpose / Category** | Data validation & leakage prevention check |
| **Affected Sections / Files** | `assignment2/src/data/dataset.py`, Report Sec. 2.3 |
| **Representative Prompt** | *"Nhóm mình chia train/val/test theo scene trong DIODE để tránh data leakage. Nhờ AI check xem logic phân chia trong `scenes_used.csv` có bị trùng lặp scene nào giữa các split không."* |
| **AI Contribution** | Verified scene ID sets across train (`00002..00015`), val (`00001`, `00017`), and test (`00000`, `00018`), confirming 0% scene overlap across all splits. |
| **Student Verification & Editing** | Ran set intersection check in Python (`set(train_scenes) & set(val_scenes)`), verified 0 common elements, and exported the final mapping table. The table is committed as `proposal/scenes_used.csv` and is regenerated from the data by `scripts/data/export_scenes.py`, which exits with an error if any scene appears in more than one split. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | Python set operations check & DIODE metadata documentation. |

---

### Log Entry 7: SILog Loss Mathematical Formulation Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Methodology (04 October 2026) |
| **Purpose / Category** | Concept explanation & mathematical formulation |
| **Affected Sections / Files** | Report Sec. 3.2 & Sec. 5 |
| **Representative Prompt** | *"Check giúp mình công thức hàm Scale-Invariant Log Loss (SILog) của Eigen et al. (2014) dùng cho Depth Estimation xem công thức LaTeX ghi trong report đã chuẩn chưa."* |
| **AI Contribution** | Provided standard mathematical notation for SILog loss equation including per-pixel valid mask $m_i$ and variance regularization term $\lambda = 0.5$. |
| **Student Verification & Editing** | Cross-referenced equation with Eigen et al. (NIPS 2014) paper and verified PyTorch implementation logic. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | Eigen et al. (2014) *"Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"* & PyTorch loss function implementation. |

---

### Log Entry 8: HTML Report Structure & Handbook Compliance Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Documentation (04 October 2026) |
| **Purpose / Category** | Report formatting & compliance check |
| **Affected Sections / Files** | `assignment2/index.html` |
| **Representative Prompt** | *"Rà soát giúp mình file index.html xem đã có đủ 16 mục bắt buộc của Assignment page theo handbook Section 18 chưa."* |
| **AI Contribution** | Highlighted missing dedicated section headers for Results, Comparison & Discussion, Error Analysis, and Links to Checkpoints/Slides/Video, and provided standard HTML section template. |
| **Student Verification & Editing** | Manually organized all HTML sections, populated M1 metrics and figures, added placeholder notes for M2/M3, and verified layout rendering locally at `http://localhost:8085`. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | CO3133 Course Handbook (Section 18 Dataset Proposal Requirements & Section 5 AI Disclosure). |

---

### Log Entry 9: Proposal Compliance Review (Section 18 and Section 21)

| Field | Record |
|---|---|
| **Tool Name & Version** | Claude Code CLI; Claude Opus 5 |
| **Member Who Used It** | Trần Hoàng Vỹ Khang |
| **Development Stage** | M1 Proposal compliance review (05 October 2026) |
| **Purpose / Category** | Compliance and architecture review against the handbook (no direct code modification) |
| **Affected Sections / Files** | `assignment2/scripts/data/export_scenes.py`, `assignment2/proposal/scenes_used.csv`, and in `assignment2/index.html`: the version/license/citation items in Section 2.1, the split-provenance and scene-manifest items in Section 2.2, the hypothesis, decision-criterion and determinism items in Section 4.1, and items 7 and 8 of Section 8.1 |
| **Representative Prompt** | *"ở phần bài tập lớn 2, tôi nên chọn dataset như thế nào"*; *"có nhánh đã push a2, hãy review xem có nên push lên main không"*; *"kiểm tra xem proposal đã đủ các mục của Section 18 và 21 chưa"*. The full session transcript is kept locally by the member and is not published. |
| **AI Contribution** | Audited the proposal against handbook Sections 18, 19.6, 20, 21 and 22 and flagged four gaps for the member to resolve: the dataset version and license were absent (Section 18 field 1), the subset selection rule reducing 26,229 images to 7,195 needed explicit documentation (field 11), `scenes_used.csv` was cited as evidence for the no-leakage claim but had not yet been committed, and the controlled experiment needed an explicit quantitative decision criterion (Section 21). Noted from the official DIODE devkit that DIODE publishes no test set and that the official validation set holds 771 images, confirming that all three splits come from the official training partition. |
| **Student Verification & Editing** | The responsible member verified the upstream MIT license on `diode-dataset.org` and `diode-dataset/diode-devkit` as well as on the Kaggle mirror (`artemmmtry/diode-a-dense-indoor-and-outdoor-depth-dataset` Version 8), updated Section 2.1, Section 2.2 and Section 4.1 of `assignment2/index.html`, and ran `scripts/data/export_scenes.py` against the dataset to generate and verify `proposal/scenes_used.csv` with zero scene overlap across splits. |
| **Responsible Member** | Trần Hoàng Vỹ Khang |
| **Verification Sources** | diode-dataset.org; `diode-dataset/diode-devkit`; Vasiljevic et al. (2019), CoRR abs/1908.00463; CO3133 Course Handbook Sections 18, 19.6, 20, 21, 22 |

---

### Log Entry 10: Subset Pipeline Review and Diagnosis of Bilinear Depth Resampling

| Field | Record |
|---|---|
| **Tool Name & Version** | Claude Code CLI; Claude Opus 5 |
| **Member Who Used It** | Trần Hoàng Vỹ Khang |
| **Development Stage** | M1 Proposal, subset reproducibility review (05 October 2026) |
| **Purpose / Category** | Data pipeline architecture review and debugging (no direct code modification) |
| **Affected Sections / Files** | `assignment2/scripts/data/make_subset.py`, `assignment2/scripts/data/kaggle_diagnose.py`, `assignment2/notebooks/monocular-depth-estimation.ipynb`, and in `assignment2/index.html`: the version item in Section 2.1, the subset selection rule in Section 2.2, the resampling items in Section 3.1, the EDA table in Section 2.4, and items 6 to 8 of Section 8.1 |
| **Representative Prompt** | *"review notebook tạo subset trên Kaggle và kiểm tra quy tắc chọn mẫu cùng phép resize depth/mask"*. The full session transcript is kept locally by the member and is not published. |
| **AI Contribution** | Reviewed the Kaggle notebook that built the subset to verify the per-domain target counts, per-scene caps, seeded shuffle order, and 70/15/15 scene-level partition. **Flagged a preprocessing defect in the initial notebook**: depth maps were being resized with `cv2.INTER_LINEAR` while the validity mask used `cv2.INTER_NEAREST`. Because DIODE encodes invalid pixels as 0, bilinear averaging pulled valid depths bordering invalid regions downward without the nearest-neighbour mask marking those boundary pixels as invalid. Suggested checking border-to-interior depth ratios and switching depth resizing to `cv2.INTER_NEAREST`. |
| **Student Verification & Editing** | The responsible member updated the subset generation pipeline (`notebooks/monocular-depth-estimation.ipynb` and `scripts/data/make_subset.py`) to resize both depth and mask with `cv2.INTER_NEAREST`, rebuilt the 7,195-sample subset on Kaggle, verified via `scripts/data/diagnose_subset.py` and `scripts/data/kaggle_diagnose.py` that mask values are strictly `{0, 1}` with border-to-interior median depth ratios of 0.81–1.03, and re-ran `scripts/data/eda.py` to update Section 2.4 of `assignment2/index.html` and `outputs/eda/summary.md`. |
| **Responsible Member** | Trần Hoàng Vỹ Khang |
| **Verification Sources** | The group's Kaggle notebook `dokhoa05/monocular-depth-estimation`; the Kaggle mirror `artemmmtry/diode-a-dense-indoor-and-outdoor-depth-dataset` Version 8; OpenCV `cv2.resize` interpolation documentation; CO3133 Course Handbook Sections 18 and 22 |

---

### Log Entry 11: M2 Loss Functions and Depth Metrics Engine

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Dương Đăng Khoa |
| **Development Stage** | M2 Draft (24 October 2026) |
| **Purpose / Category** | Implementation review, edge-case validation, and test suite design |
| **Affected Sections / Files** | `assignment2/src/engine/loss.py`, `assignment2/src/engine/metrics.py`, `assignment2/tests/test_metrics.py` (originally `scripts/test_loss_metrics.py`; files later revised and moved, see entry 13) |
| **Representative Prompt** | *"Check giúp mình logic cài đặt MaskedSILogLoss: khi batch gặp toàn bộ mask 0 thì return pred.sum() * 0.0 đã đủ để giữ computational graph cho backward pass chưa, và gợi ý thêm test case kiểm tra tính chất scale-invariance khi lambda=1.0."* |
| **AI Contribution** | Reviewed PyTorch autograd graph preservation for empty masks; verified SILog loss scale-invariance properties; assisted in formulating edge-case assertion checks (NaNs, Infs, zero-division). |
| **Student Verification & Editing** | Implemented `MaskedSILogLoss`, `compute_depth_metrics`, and `DepthMetricTracker`; authored and ran `scripts/test_loss_metrics.py` verifying gradient flow and metric accuracy across 4 test suites. |
| **Responsible Member** | Dương Đăng Khoa |
| **Verification Sources** | Eigen et al. (NeurIPS 2014); PyTorch Autograd documentation; CO3133 Course Handbook Section 20.1. |

---
### Log Entry 12: Baseline U-Net Architecture and Training Pipeline

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Dương Đăng Khoa |
| **Development Stage** | M2 Draft (24 October 2026) |
| **Purpose / Category** | Architecture design review, training pipeline implementation, and end-to-end verification |
| **Affected Sections / Files** | `assignment2/src/models/unet.py`, `assignment2/src/engine/trainer.py`, `assignment2/configs/baseline_unet.yaml`, `assignment2/train.py`, `assignment2/tests/test_models.py`, `assignment2/tests/test_pipeline.py` (originally under `scripts/`; files later revised and moved, see entry 13) |
| **Representative Prompt** | *"Rà soát giúp mình kiến trúc U-Net 4 tầng với skip-connections cho bài toán Depth (288x384) và pipeline huấn luyện trong `trainer.py`: kiểm tra cơ chế GradScaler AMP fp16, tiêu chí lưu best checkpoint theo Abs Rel và thiết kế script test giả lập end-to-end."* |
| **AI Contribution** | Reviewed U-Net encoder-decoder channel sizing (32 to 512) and positive depth output activation; assisted in configuring PyTorch AMP GradScaler and metric tracking integration; provided synthetic end-to-end pipeline verification test. |
| **Student Verification & Editing** | Self-implemented UNet, Trainer, YAML configuration, and training CLI script; executed `test_unet.py` (7.76M params verified) and `test_pipeline_synthetic.py` confirming loss convergence, history CSV logging, and checkpoint serialization. |
| **Responsible Member** | Dương Đăng Khoa |
| **Verification Sources** | Ronneberger et al. (MICCAI 2015); PyTorch AMP documentation; CO3133 Course Handbook Section 20. |

---
### Log Entry 13: M2 Architecture Control, Code Review and Evaluation Verification

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Claude model) |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M2 Draft (26 October 2026) |
| **Purpose / Category** | Architecture control, code review of entries 11–12, and diagnostic verification (no direct code modification) |
| **Affected Sections / Files** | `src/engine/{metrics,align,loss,trainer}.py`, `src/models/{unet,midas,__init__}.py`, `src/utils/common.py`, `src/data/dataset.py`, `configs/*.yaml`, `train.py`, `evaluate.py`, `run_all.sh`, `tests/`, `scripts/report/`, `README.md`, `index.html` Sec. 4–8. |
| **Representative Prompt** | *"Đọc kĩ src code, review kiến trúc và kiểm tra xem pipeline M2 có điểm nào chưa chặt chẽ hoặc lệch yêu cầu handbook không."* |
| **AI Contribution** | Reviewed the M2 codebase and flagged architectural and evaluation issues for the member to fix: (1) metric calculation previously dropped pixels with `pred <= 0` instead of clamping them, which would flatter models predicting non-positive values; (2) U-Net output activation needed `exp(clamp(.))` on log-depth rather than `ReLU` to avoid zero-gradient dead zones; (3) frozen-encoder MiDaS required `self.net.pretrained.eval()` inside `.train()` so BatchNorm running statistics stay fixed alongside frozen weights; (4) MiDaS relative inverse depth required scale-and-shift-invariant (SSI) loss and per-image least-squares disparity alignment (`s * pred + t ~ 1/GT`); and (5) sub-0.6 m outdoor scanner noise (below the FARO S350 0.6 m minimum range) was distorting relative error on foliage/glass boundaries. |
| **Student Verification & Editing** | The responsible member implemented the architectural fixes, reorganized the data/report/test folders, wrote and ran the unit test suite (`tests/test_metrics.py`, `tests/test_models.py`, `tests/test_pipeline.py`), verified deterministic reproducibility across repeated smoke runs, and executed all four training/evaluation configurations (`baseline_unet`, `midas_zeroshot`, `midas_full`, `midas_frozen`) plus the boundary-erosion (`erode_check.py`) and per-scene (`outdoor_check.py`) diagnostics on the group RTX 4060 GPU. All reported numbers and figures in `results/` and `index.html` were verified directly against the executed run logs. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | Ranftl et al. (TPAMI 2020, MiDaS SSI loss and alignment); Eigen et al. (2014); Vasiljevic et al. (2019, DIODE FARO S350 sensor range); PyTorch reproducibility and AMP docs. |

---

## Declaration of Ownership

All code, data preprocessing choices, baseline and pretrained model pipelines, statistical findings, and conclusions presented in this assignment are written, verified, and owned by group CR7. AI tools were used strictly for code review, architecture control, mathematical/compliance verification and debugging diagnostics, not for direct code modification or as a replacement for original work. No reported number was produced by anything other than an executed run on group hardware.

Every file and analysis in this repository has been reviewed and verified line by line by the responsible group members. The group accepts that the instructor may interview any member and request explanation, live edits, metric interpretation or partial reproduction (handbook Section 5.5).
