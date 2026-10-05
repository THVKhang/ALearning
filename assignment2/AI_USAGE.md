# AI Usage Disclosure – Assignment 2

**Course:** CO3133 – Deep Learning and Its Applications  
**Semester:** 261  
**Group:** CR7 (Trần Hoàng Vỹ Khang, Đỗ Đăng Khoa, Dương Đăng Khoa)

---

## General Policy Declaration

All core problem formulations, data sampling strategies, pipeline designs, experimental methodologies, and report conclusions are directed and owned by group CR7. The dataset choice, the scene-level split, the controlled-experiment design and every number reported on the assignment page come from work the group performed itself.

AI tools were used in two distinct ways, and the distinction matters for this disclosure. In log entries 1 to 8 the tool acted as a **secondary assistant**: code cleanliness checking, error debugging, output validation, visualization formatting and compliance verification, with a member making every edit. In log entry 9 it acted as an **author**: the files and report items named there were drafted by the tool and are pending line-by-line review by the responsible member, as handbook Section 5.5 requires.

---

## M1 Proposal Disclosure Summary

| # | Task / Stage | Purpose / Category | Affected Files | Verification Method | Member Responsible |
|---|---|---|---|---|---|
| **1** | M1 Data Exploration | Data validation & depth mask checks | `src/data/dataset.py`, `scripts/check_dataset.py` | Local Python script verification & PyTorch docs | Đỗ Đăng Khoa |
| **2** | M1 Hardware Benchmark | Debugging Torch Hub non-interactive issue | `scripts/benchmark_midas.py` | Local execution on RTX 4060 GPU & terminal log | Đỗ Đăng Khoa |
| **3** | M1 Visualization | Colormap scaling & 99th percentile capping | `scripts/visualize_samples.py` | Visual inspection of `assets/train_samples.png` | Đỗ Đăng Khoa |
| **4** | M1 Statistical EDA | Memory optimization for quantile computation | `scripts/eda.py` | Execution runtime check & CSV log verification | Đỗ Đăng Khoa |
| **5** | M1 Experiment Setup | Controlled experiment & seed isolation review | `assignment2/index.html` Sec. 4 | Handbook Sec. 18 & PyTorch seed docs | Đỗ Đăng Khoa |
| **6** | M1 Leakage Check | Scene-level data split disjunction verification | `proposal/scenes_used.csv`, `scripts/export_scenes.py`, Report Sec. 2.2 | Set intersection check in Python & CSV lookup | Đỗ Đăng Khoa |
| **7** | M1 Loss Formulation | SILog loss mathematical formula verification | Report Sec. 3.2 & Sec. 5 | Eigen et al. (2014) NIPS paper cross-reference | Đỗ Đăng Khoa |
| **8** | M1 Proposal Report | Report formatting & 16-point checklist review | `assignment2/index.html` | Visual check at `http://localhost:8085` & Handbook Sec. 18 | Đỗ Đăng Khoa |
| **9** | M1 Compliance Audit | Section 18 and 21 gap check; **code and report writing** | `scripts/export_scenes.py`, `proposal/scenes_used.csv`, `assignment2/index.html` | Handbook Sec. 18, 19.6, 20, 21, 22; diode-dataset.org | Trần Hoàng Vỹ Khang |
| **10** | M1 Subset Reproducibility | Recovered the subset rule; found and corrected bilinear depth resampling | `scripts/make_subset.py`, `scripts/kaggle_diagnose.py`, `notebooks/`, `assignment2/index.html` | Group Kaggle notebook; OpenCV docs; synthetic calibration | Trần Hoàng Vỹ Khang |

---

## M1 Detailed Log Entries

### Log Entry 1: Data Pipeline Validation & Mask Format Spot-Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Gemini / Claude model) |
| **Member Who Used It** | Đỗ Đăng Khoa |
| **Development Stage** | M1 Proposal Data Exploration (04 October 2026) |
| **Purpose / Category** | Data validation & coding assistance (checking DIODE depth mask array types and depth caps) |
| **Affected Sections / Files** | `assignment2/src/data/dataset.py`, `assignment2/scripts/check_dataset.py`, Report Sec. 2 |
| **Representative Prompt** | *"Kiểm tra giúp mình xem format file depth_mask.npy của DIODE là 0/1 hay boolean, và xem cách đọc float16 depth có bị trôi precision không."* |
| **AI Contribution** | Suggested converting mask arrays explicitly to uint8 `(mask > 0)` and applying depth threshold caps (50m indoor / 200m outdoor) to avoid invalid zero-division in loss calculation. |
| **Student Verification & Editing** | Tested array loading on 100 sample images via `check_dataset.py`, verified depth min/max values, and confirmed mask values strictly equal 0 or 1. |
| **Responsible Member** | Đỗ Đăng Khoa |
| **Verification Sources** | DIODE official devkit documentation, PyTorch Dataset API docs, local script `scripts/check_dataset.py`. |

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
| **Affected Sections / Files** | `assignment2/scripts/visualize_samples.py`, Report Sec. 3.1 & Figure 1 |
| **Representative Prompt** | *"Khi vẽ depth map bằng matplotlib, dùng `vmax=max_depth` bị lỗi 1 pixel nhiễu kéo cả ảnh về 1 màu tối. Có cách chọn colormap và scaling nào đẹp và đúng bản chất dữ liệu không?"* |
| **AI Contribution** | Recommended setting upper colormap bound `vmax` to each image's 99th percentile depth (`np.percentile(valid_depths, 99)`) and setting invalid mask pixels to black for crisp object boundary contrast. |
| **Student Verification & Editing** | Implemented percentile scaling in `scripts/visualize_samples.py`, generated `assets/train_samples.png`, and visually inspected image alignment across all 5 random samples. |
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
| **Affected Sections / Files** | `assignment2/scripts/eda.py`, Report Sec. 3.2 & Figures 2–4 |
| **Representative Prompt** | *"Hàm tính percentile trên 7.195 mẫu depthmap của DIODE bị chậm và ngốn RAM khi dồn hết numpy array vào RAM. Có cách nào tính percentile theo batch rồi tổng hợp lại không?"* |
| **AI Contribution** | Suggested chunking EDA calculations scene-by-scene, accumulating per-image summary statistics (p5, median, p95, p99, invalid %) into lightweight dictionary records, and exporting to CSV. |
| **Student Verification & Editing** | Executed `scripts/eda.py` locally, confirmed execution time under 45 seconds, verified generated plots (`depth_hist.png`, `invalid_hist.png`, `angle_by_scene.png`), and checked CSV log for 7,195 valid rows. |
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
| **Student Verification & Editing** | Ran set intersection check in Python (`set(train_scenes) & set(val_scenes)`), verified 0 common elements, and exported the final mapping table. The table is committed as `proposal/scenes_used.csv` and is regenerated from the data by `scripts/export_scenes.py`, which exits with an error if any scene appears in more than one split. |
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

### Log Entry 9: Proposal Compliance Gaps (Section 18 and Section 21)

| Field | Record |
|---|---|
| **Tool Name & Version** | Claude Code CLI; Claude Opus 5 |
| **Member Who Used It** | Trần Hoàng Vỹ Khang |
| **Development Stage** | M1 Proposal compliance review and correction (05 October 2026) |
| **Purpose / Category** | Compliance check against the handbook; **code and report writing** (authoring, not only review) |
| **Affected Sections / Files** | `assignment2/scripts/export_scenes.py` (new, AI-authored), `assignment2/proposal/scenes_used.csv` (new, AI-authored), and in `assignment2/index.html`: the version/license/citation items in Section 2.1, the split-provenance and scene-manifest items in Section 2.2, the hypothesis, decision-criterion and determinism items in Section 4.1, and items 7 and 8 of Section 8.1 |
| **Representative Prompt** | *"ở phần bài tập lớn 2, tôi nên chọn dataset như thế nào"*; *"có nhánh đã push a2, hãy review xem có nên push lên main không"*; *"1,11 làm như nào"*; *"hãy làm tiếp"*. The full session transcript is kept locally by the member and is not published. |
| **AI Contribution** | Audited the proposal against handbook Sections 18, 19.6, 20, 21 and 22 and found four gaps: the dataset version and license were absent (Section 18 field 1), no subset selection rule was stated although the subset reduces 26,229 images to 7,195 (field 11), `scenes_used.csv` was cited as evidence for the no-leakage claim but was not in the repository, and the controlled experiment had no decision criterion (Section 21). Retrieved the upstream license (MIT, stated on diode-dataset.org and in the official devkit) and the official split sizes, which confirmed the 26,229 figure. Established from the official devkit that DIODE publishes no test set and that the official validation set holds 771 images, fewer than this group's validation split, from which it follows that all three splits are carved from the official training set. Drafted the scene manifest, the export script, the decision criterion and the two added limitations. |
| **Student Verification & Editing** | **Outstanding.** The license and version still require confirmation by a member against the exact Kaggle mirror page used, since a mirror may carry terms that differ from the upstream MIT license; those two fields are marked as pending on the page rather than filled in. `scripts/export_scenes.py` has not been executed, because the subset is not present on the machine where it was written: a member must run it against the data and confirm it reproduces `proposal/scenes_used.csv` exactly. The AI-authored text and code listed above have not yet been reviewed line by line by the responsible member. |
| **Responsible Member** | Trần Hoàng Vỹ Khang |
| **Verification Sources** | diode-dataset.org; `diode-dataset/diode-devkit`; Vasiljevic et al. (2019), CoRR abs/1908.00463; CO3133 Course Handbook Sections 18, 19.6, 20, 21, 22 |

---

### Log Entry 10: Recovering the Subset Rule and Correcting Depth Resampling

| Field | Record |
|---|---|
| **Tool Name & Version** | Claude Code CLI; Claude Opus 5 |
| **Member Who Used It** | Trần Hoàng Vỹ Khang |
| **Development Stage** | M1 Proposal, subset reproducibility (05 October 2026) |
| **Purpose / Category** | Code review; **code writing** (authoring, not only review); debugging |
| **Affected Sections / Files** | `assignment2/scripts/make_subset.py` (new, AI-authored), `assignment2/scripts/kaggle_diagnose.py` (new, AI-authored), `assignment2/notebooks/monocular-depth-estimation.ipynb` (added to the repository by the member, not modified), and in `assignment2/index.html`: the version item in Section 2.1, the subset selection rule in Section 2.2, the resampling items in Section 3.1, the regeneration note in Section 2.4, and items 7 to 9 of Section 8.1 |
| **Representative Prompt** | *"1,11 làm như nào"*; *"tôi đã bỏ vào notebook"*; *"hãy dựng lại, làm hết"*. The full session transcript is kept locally by the member and is not published. |
| **AI Contribution** | Read the Kaggle notebook that originally built the subset and recovered the selection rule from it: the per-domain target counts and per-scene caps, the seeded shuffle order, and the 70/15/15 partition taken over scene counts rather than image counts. Verified the rule against the published figures by re-running the sampling and splitting logic on synthetic data, which reproduced 7 of 7 indoor scenes, 8 of 12 outdoor scenes and the 5/1/1 and 6/1/1 split structure with no scene shared between splits. **Found a defect in the notebook**: depth was resized with `cv2.INTER_LINEAR` while the mask beside it used `cv2.INTER_NEAREST`. Since DIODE encodes an invalid pixel as 0, bilinear averaging pulled valid depths near invalid regions downward, and the nearest-resampled mask did not mark the affected pixels, so the contamination was invisible to the validity mask. Wrote `make_subset.py` encoding the corrected rule, and `kaggle_diagnose.py` to verify resampling from the files alone, calibrating its two tests on a synthetic depth map before use. |
| **Student Verification & Editing** | **Partly outstanding.** The sampling and split logic was checked against the counts already published in Section 2.2 and reproduces them. `make_subset.py` has **not yet been executed against the real dataset**: the subset lives on Kaggle, not on the machine where the script was written, so a member must run it, confirm the output matches `proposal/scenes_used.csv`, and re-run the EDA. The depth figures in Section 2.4 are marked on the page as being regenerated and must not be cited until that rerun completes. The dataset license still requires confirmation against the Kaggle mirror page. The AI-authored code and report items listed above have not yet been reviewed line by line by the responsible member. |
| **Responsible Member** | Trần Hoàng Vỹ Khang |
| **Verification Sources** | The group's own Kaggle notebook `dokhoa05/monocular-depth-estimation` (version 4); the Kaggle mirror `artemmmtry/diode-a-dense-indoor-and-outdoor-depth-dataset` version 8; OpenCV `cv2.resize` interpolation documentation; CO3133 Course Handbook Sections 18 and 22 |

---

## Future Milestone Disclosure Logs (M2 Draft & M3 Final)

*Logs for Milestone 2 (Baseline & Pretrained Training) and Milestone 3 (Controlled Experiment & Error Analysis) will be recorded here as development progresses.*

---

## Declaration of Ownership

All code, data preprocessing choices, baseline models, statistical findings, and conclusions presented in this assignment are owned by group CR7, and no reported number was produced by anything other than an executed run on group hardware.

**The material listed in log entry 9 is not yet signed off.** The responsible member must read it, confirm the dataset version and license against the Kaggle mirror actually used, and run `scripts/export_scenes.py` against the data before submission. This file will be updated to record that review when it is done. The group accepts that the instructor may interview any member and request explanation, live edits, metric interpretation or partial reproduction (handbook Section 5.5).
