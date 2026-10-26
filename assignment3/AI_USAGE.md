# AI Usage Disclosure – Assignment 3

**Course:** CO3133 – Deep Learning and Its Applications  
**Semester:** 261  
**Group:** Nhóm CR7 (Trần Hoàng Vỹ Khang – `2352502`, Đỗ Đăng Khoa – `2352554`, Dương Đăng Khoa – `2352551`)  
**Assignment:** 3 – Multimodal Deep Learning: Representation, Fusion, and Evaluation

This file is the detailed log required by handbook Section 5.1, item 3. Summary versions appear on the [shared landing page](../index.html) (Section 5.1, item 1), on the [Assignment 3 page](index.html) (Section 5.1, item 2), and in the M3 Final Report (`CR7_A3_Report.md`). All nine fields required by Section 5.2 are recorded for every entry below.

---

## General Policy Declaration

All core problem formulations, cross-modal pair-validity checks, scene-level split designs, unimodal and multimodal architectures, ablation studies, support/conflict analyses, and report conclusions are written, executed, and owned by group CR7.

Across all stages of Assignment 3, AI tools acted strictly as a **code reviewer, architecture controller, and consistency auditor**, not as a direct code modifier or autonomous author: reviewing member-written files for cleanliness and edge cases, controlling encoder and fusion tensor shapes, checking modality-dropout and validity-gating behavior under degraded LiDAR returns, and cross-checking reported tables against the committed JSON run artifacts. Every edit, training run, and verification step was performed directly by the responsible group members, as handbook Section 5.5 requires.

---

## Summary of Disclosure Entries

| # | Stage / Task | Purpose / Category (Section 5.2) | Affected Files / Sections | Verification Method | Responsible Member |
|---|---|---|---|---|---|
| **1** | M1 Multimodal Proposal & EDA | Pair-validity check, scene-disjoint split verification, and 6-class balance review | `src/data/dataset.py`, `scripts/eda.py`, `proposal/scenes_used.csv`, `outputs/eda/summary.md` | 100% pair-completeness check (7,195 pairs); scene set-disjointness assertion | Trần Hoàng Vỹ Khang (`2352502`) |
| **2** | M2 Unimodal & Simple Fusion Baselines | Architecture control & code review of Modality A (`rgb_only`), Modality B (`depth_only`), and `simple_fusion` | `src/models/encoders.py`, `src/fusion/modules.py`, `src/models/multimodal.py`, `tests/test_multimodal.py` | Unit tests (`test_multimodal.py`); shape & gradient flow verification | Dương Đăng Khoa (`2352551`) |
| **3** | M3 Improved Fusion & Ablation Design | Architecture review of bidirectional cross-modal attention, LiDAR validity-conditioned gate, auxiliary loss, and modality dropout | `src/fusion/modules.py`, `src/models/multimodal.py`, `src/engine/loss.py`, `configs/*.yaml` | Controlled ablations (`ablation_no_gate`, `ablation_no_aux`) and missing-modality stress tests | Dương Đăng Khoa (`2352551`) |
| **4** | M3 Support/Conflict & Error Audit | Review of cross-modal support/conflict taxonomy, missing-modality evaluation, and report/web numerical consistency | `evaluate.py`, `scripts/make_report.py`, `README.md`, `index.html`, `CR7_A3_Report.md` | Cross-checking every table cell against `outputs/*/eval_test.json` and `support_conflict_test.json` | Đỗ Đăng Khoa (`2352554`) |

---

## Detailed Log Entries (All 9 Mandatory Fields of Section 5.2)

### Entry 1 — Cross-Modal Pair Validity and Scene-Disjoint Split Review (M1 Proposal & EDA)

| Field | Record |
|---|---|
| **Tool Name & Version** | Claude Code CLI / Antigravity AI Assistant (Claude Sonnet 5, Claude Opus 5) |
| **Member Who Used It** | Trần Hoàng Vỹ Khang (`2352502`) |
| **Time / Development Stage** | Oct 2026; M1 Proposal and Multimodal EDA verification |
| **Purpose / Category** | Data pipeline review and cross-modal compliance verification (Handbook Sections 27, 28.1, 28.2, 31) |
| **Affected Sections / Files** | `assignment3/src/data/dataset.py`, `assignment3/scripts/eda.py`, `assignment3/proposal/scenes_used.csv`, `assignment3/outputs/eda/summary.md`, Report Part 1 |
| **Representative Prompt** | *"Kiểm tra thiết kế bài toán phân loại đa phương thức 6 lớp (Domain x Camera Elevation) trên 7,195 cặp RGB-Depth của DIODE theo đúng các ràng buộc Section 28.1 và 28.2 của handbook."* |
| **AI Contribution** | Reviewed the 6-class formulation (`{Indoor, Outdoor} x {Level 000–010°, Mid 020–030°, High 040–050°}`) against Handbook Section 28.1 and 28.2 thresholds (confirming ≥ 5 classes and ≥ 5,000 genuinely co-registered pairs from the same physical scanner mount and timestamp); checked that synchronous horizontal flip augmentation preserves both the domain and vertical elevation labels; and verified that the 3-channel geometric representation (`[log_depth, valid_mask, grad_y]`) explicitly isolates invalid laser returns (`mask = 0`) rather than mixing zero depth into vertical gradient differences. |
| **How Output Was Edited & Verified** | The responsible member executed `scripts/eda.py` across all 7,195 samples, confirming 0 missing files, 0% scene overlap across `train` (11 scenes, 4,995 pairs), `val` (2 scenes, 1,100 pairs), and `test` (2 scenes, 1,100 pairs), and verified the generated class-balance and LiDAR-validity distributions in `proposal/scenes_used.csv` and `outputs/eda/summary.md`. |
| **Responsible Member** | Trần Hoàng Vỹ Khang (`2352502`) |
| **Verification Sources** | Course Handbook Sections 27, 28.1, 28.2, and 31; Vasiljevic et al. (2019, DIODE dataset specification); executed output in `assignment3/outputs/eda/summary.md`. |

---

### Entry 2 — Architecture Control of Unimodal Encoders and Simple Late Fusion (M2 Draft)

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Claude model) |
| **Member Who Used It** | Dương Đăng Khoa (`2352551`) |
| **Time / Development Stage** | Oct 2026; M2 Unimodal and Simple Fusion implementation review |
| **Purpose / Category** | Architecture review and test verification (no direct code modification) |
| **Affected Sections / Files** | `assignment3/src/models/encoders.py`, `assignment3/src/fusion/modules.py`, `assignment3/src/models/multimodal.py`, `assignment3/tests/test_multimodal.py`, Report Part 2 |
| **Representative Prompt** | *"Rà soát kiến trúc ModalityEncoder cho nhánh RGB (Modality A) và nhánh Depth-Geometry (Modality B) cùng SimpleConcatFusion để đảm bảo so sánh công bằng về số chiều embedding và số tham số."* |
| **AI Contribution** | Reviewed the symmetric 4-stage `ModalityEncoder` (`3 -> 32 -> 64 -> 96 -> 128`) used by both `rgb_only` (`463,974` params) and `depth_only` (`463,974` params), confirming identical backbone capacity so performance differences reflect modality information rather than encoder size; verified that `SimpleConcatFusion` (`[z_rgb ; z_depth] -> Linear(256, 128) -> LayerNorm -> GELU -> Dropout`) returns raw unnormalized 6-class logits for `nn.CrossEntropyLoss`. |
| **How Output Was Edited & Verified** | The responsible member ran `tests/test_multimodal.py` to verify forward output shapes `(B, 6)` and non-zero backward gradients across all four model families, then trained and evaluated `rgb_only`, `depth_only`, and `simple_fusion` on the group GPU. |
| **Responsible Member** | Dương Đăng Khoa (`2352551`) |
| **Verification Sources** | PyTorch `nn.Module`, `nn.MultiheadAttention`, and `nn.CrossEntropyLoss` documentation; Course Handbook Section 29.1 & 29.2. |

---

### Entry 3 — Review of Cross-Modal Attention, Validity-Conditioned Gating, and Ablation Design (M3 Final)

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Claude model) |
| **Member Who Used It** | Dương Đăng Khoa (`2352551`) |
| **Time / Development Stage** | Oct 2026; M3 Improved Fusion and Ablation architecture control |
| **Purpose / Category** | Architecture review and controlled experiment design check (Handbook Sections 29.2 & 30) |
| **Affected Sections / Files** | `assignment3/src/fusion/modules.py`, `assignment3/src/models/multimodal.py`, `assignment3/src/engine/loss.py`, `assignment3/configs/*.yaml`, Report Parts 2 & 3 |
| **Representative Prompt** | *"Kiểm tra cơ chế BidirectionalCrossModalAttention trên 12 vùng không gian (3x4) kết hợp cổng tin cậy ValidityConditionedGatedFusion nhận tỉ lệ valid_ratio của LiDAR và hàm loss phụ trợ unimodal."* |
| **AI Contribution** | Reviewed the intermediate token-level cross-attention (`12` spatial tokens of dimension `128` per modality with learnable positional encoding) and the reliability gate $g = \sigma(\text{MLP}([z_{\text{rgb}}, z_{\text{depth}}, r_{\text{valid}}]))$; verified that stochastic modality dropout ($p_{\text{drop}} = 0.15$) is active only when `self.training == True`; and checked that the two controlled ablations (`ablation_no_gate` and `ablation_no_aux`) isolate the spatial attention/validity gate and the auxiliary supervision/dropout respectively while holding seed, optimizer, epochs, and data splits fixed. |
| **How Output Was Edited & Verified** | Verified gate activation values on clean ($\ge 65\%$ valid LiDAR) vs. noisy ($< 65\%$ valid LiDAR) test samples and executed the missing-modality stress tests (`drop_modality="depth"` and `drop_modality="rgb"`) in `evaluate.py`. |
| **Responsible Member** | Dương Đăng Khoa (`2352551`) and Đỗ Đăng Khoa (`2352554`) |
| **Verification Sources** | Course Handbook Sections 29.2 and 30; executed JSON artifacts in `assignment3/outputs/`. |

---

### Entry 4 — Review of Support/Conflict Taxonomy and Cross-Artifact Consistency (M3 Final)

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant (Claude model) |
| **Member Who Used It** | Đỗ Đăng Khoa (`2352554`), Trần Hoàng Vỹ Khang (`2352502`), and Dương Đăng Khoa (`2352551`) |
| **Time / Development Stage** | Oct 2026; M3 Final evaluation, support/conflict analysis, and documentation audit |
| **Purpose / Category** | Result analysis review and numerical consistency check across `outputs/`, `README.md`, `index.html`, and `CR7_A3_Report.md` |
| **Affected Sections / Files** | `assignment3/evaluate.py`, `assignment3/scripts/make_report.py`, `assignment3/results/tables.md`, `assignment3/README.md`, `assignment3/index.html`, `assignment3/CR7_A3_Report.md` |
| **Representative Prompt** | *"Kiểm tra script phân loại 4 nhóm đồng thuận/xung đột giữa RGB-only, Depth-only và Fusion trên tập test (1,100 mẫu), và đối chiếu toàn bộ bảng biểu trong README, index.html, CR7_A3_Report.md với JSON thực nghiệm."* |
| **AI Contribution** | Reviewed the partition of the 1,100 test predictions into four mutually exclusive regimes (`Both Correct`, `Conflict: RGB Correct / Depth Wrong`, `Conflict: Depth Correct / RGB Wrong`, and `Emergent Support: Both Unimodal Wrong`) and checked that the counts sum to exactly 1,100; verified that every metric in `README.md`, `index.html`, and `CR7_A3_Report.md` matches `outputs/*/eval_test.json` and `train_summary.json`. |
| **How Output Was Edited & Verified** | All three group members inspected the generated confusion matrices (`assets/confusion_matrices.png`), learning curves (`assets/curves_all_models.png`), and qualitative support/conflict examples (`assets/support_conflict_examples.png`) against the raw test predictions. |
| **Responsible Member** | Đỗ Đăng Khoa (`2352554`), Trần Hoàng Vỹ Khang (`2352502`), and Dương Đăng Khoa (`2352551`) |
| **Verification Sources** | Committed JSON artifacts under `assignment3/outputs/`; Course Handbook Sections 3, 5.2, 29–32. |

---

## Declaration of Ownership

All code, multimodal data pipelines, encoder and fusion architectures, experimental runs, and scientific conclusions in Assignment 3 are written, verified, and owned by group CR7. AI tools were used strictly for code review, architecture control, and consistency verification, not for direct code modification or as a replacement for original work. Every reported number comes from an executed run on the group hardware.

The group accepts that the instructor may interview any member and request explanation, live edits, metric interpretation, or partial reproduction (handbook Section 5.5).
