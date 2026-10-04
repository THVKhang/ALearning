# AI Usage Disclosure – Assignment 2

**Course:** CO3133 – Deep Learning and Its Applications  
**Semester:** 261  
**Group:** CR7 (Trần Hoàng Vỹ Khang, Đỗ Đăng Khoa, Dương Đăng Khoa)

---

## General Policy Declaration

All core problem formulations, data sampling strategies, pipeline designs, code implementations, experimental methodologies, and report conclusions are completely directed, implemented, and owned by group CR7. AI tools were utilized strictly as secondary assistants for code cleanliness checking, error debugging, output validation, and compliance verification under direct human supervision.

---

## M1 Proposal Disclosure Logs

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
| **Member Who Used It** | Trần Hoàng Vỹ Khang |
| **Development Stage** | M1 Proposal Hardware Benchmarking (04 October 2026) |
| **Purpose / Category** | Debugging & performance measurement |
| **Affected Sections / Files** | `assignment2/scripts/benchmark_midas.py`, Report Sec. 4 & Sec. 7 |
| **Representative Prompt** | *"Script benchmark MiDaS bị treo khi gọi torch.hub.load('intel-isl/MiDaS', 'MiDaS_small') trong môi trường non-interactive, cách fix như thế nào?"* |
| **AI Contribution** | Identified that `torch.hub.load` was prompting for interactive confirmation when downloading dependencies (EfficientNet backbone repo `rwightman/gen-efficientnet-pytorch`) and recommended passing `trust_repo=True`. |
| **Student Verification & Editing** | Updated `benchmark_midas.py` with `trust_repo=True`, ran full 100-iter benchmark on local NVIDIA RTX 4060 GPU, and verified VRAM and latency output logs. |
| **Responsible Member** | Trần Hoàng Vỹ Khang |
| **Verification Sources** | PyTorch `torch.hub` official documentation & local GPU execution terminal log. |

---

### Log Entry 3: HTML Report Structure & Handbook Compliance Check

| Field | Record |
|---|---|
| **Tool Name & Version** | Antigravity AI Assistant |
| **Member Who Used It** | Dương Đăng Khoa |
| **Development Stage** | M1 Proposal Documentation (04 October 2026) |
| **Purpose / Category** | Report formatting & compliance check |
| **Affected Sections / Files** | `assignment2/index.html` |
| **Representative Prompt** | *"Rà soát giúp mình file index.html xem đã có đủ 16 mục bắt buộc của Assignment page theo handbook Section 18 chưa."* |
| **AI Contribution** | Highlighted missing dedicated section headers for Results, Comparison & Discussion, Error Analysis, and Links to Checkpoints/Slides/Video, and provided standard HTML section template. |
| **Student Verification & Editing** | Manually organized all HTML sections, populated M1 metrics and figures, added placeholder notes for M2/M3, and verified layout rendering locally at `http://localhost:8085`. |
| **Responsible Member** | Dương Đăng Khoa |
| **Verification Sources** | CO3133 Course Handbook (Section 18 Dataset Proposal Requirements & Section 5 AI Disclosure). |

---

## Declaration of Ownership

All code, data preprocessing choices, baseline models, statistical findings, and conclusions presented in this assignment are fully understood, verified, and owned by group CR7.
