# AI Usage Disclosure - Assignment 1

Required by the course handbook, Section 5. This file is the detailed log; the summary
versions belong on the landing page, on the Assignment 1 page, and in the report.

> **ACTION REQUIRED BEFORE SUBMISSION.** The fields marked `TODO` must be completed by
> the group. Nothing here may be submitted until the responsible member has read the
> code, can explain every part of it, and has signed off on the verification rows. The
> handbook makes the students - not the tool - responsible for all submitted code and
> numbers.

## Tool 1 - Claude Code (Anthropic Claude)

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Sonnet 5, then Claude Opus 5 (same session) |
| Used by | TODO - confirm member name (session ran under git user `GabrielHCMUT`) |
| Stage | Environment setup and A1-M1 Draft implementation, 16 Sep 2026 |
| Purpose | See the itemized list below |
| Responsible member | TODO |
| Verification sources | PyTorch and torchvision documentation, scikit-learn documentation, executed runs and their artifacts in `outputs/` |

### What the tool was asked to do

1. **Environment repair.** Diagnosed `torch.cuda.is_available() == False`: the virtual
   environment had the CPU-only wheel `torch 2.14.0+cpu` while the machine has an RTX
   3050 with a CUDA 13.4 driver. Reinstalled `torch 2.14.0+cu130` and
   `torchvision 0.29.0+cu130` from PyTorch's CUDA index, then installed `numpy`,
   `scikit-learn`, `matplotlib`, `PyYAML`, and the notebook packages.
   - *Prompt summary:* "why is torch.cuda.is_available() False / uninstall and reinstall".
   - *Verified by:* `torch.cuda.is_available()` returns `True` and reports
     `NVIDIA GeForce RTX 3050 Laptop GPU`; `nvidia-smi` confirms the driver.
2. **Repository scaffolding.** Created the `assignment1/`, `assignment2/`,
   `assignment3/` directory trees and the root `.gitignore`.
3. **Debugging student-written code.** Located a typo (`torch._version_` instead of
   `torch.__version__`), an unterminated f-string and a wrong indentation level in
   `src/utils/seed.py`, and explained a `ModuleNotFoundError` caused by running Python
   from the repository root instead of `assignment1/`. The fixes were typed by the
   student.
4. **Implementation of the draft pipeline** (files listed in the table below).
5. **EDA notebook.** Authored and executed `notebooks/eda.ipynb`.

### Files affected

| File | AI contribution | Student contribution |
|---|---|---|
| `src/utils/seed.py` | Reviewed; pointed out the syntax and indentation errors | **Written by the student** |
| `src/data/dataset.py` | Extended to add normalization, stratified splitting, and the `info` dict of split statistics | **First version written by the student** (`get_dataloaders` with `random_split`) |
| `src/utils/metrics.py` | Written by AI | TODO - review |
| `src/utils/plots.py` | Written by AI | TODO - review |
| `src/models/linear.py` | Written by AI | TODO - review |
| `src/models/mlp.py` | Written by AI | TODO - review |
| `src/engine/trainer.py` | Written by AI | TODO - review |
| `scripts/train.py` | Written by AI | TODO - review |
| `scripts/eval.py` | Written by AI | TODO - review |
| `configs/default.yaml` | Written by AI | TODO - review |
| `notebooks/eda.ipynb` | Written and executed by AI | TODO - review |
| `README.md` | Written by AI | TODO - review |

### Verification already performed inside the session

These checks were run and their evidence exists in the repository; they are *not* a
substitute for the group's own review.

- `scripts/train.py` and `scripts/eval.py` were executed end to end; all reported
  numbers in `README.md` come from those runs, not from estimates.
- The normalization constants in `src/data/dataset.py` (mean 0.2860, std 0.3530) were
  checked against the statistics measured over the 60,000 training images in
  `notebooks/eda.ipynb`; they match to four decimals.
- The stratified split was checked to yield exactly 5,400 train / 600 validation images
  per class, and one preprocessed batch was confirmed to have mean ~0 and std ~1.
- The confusion matrix and the error-example figures were rendered and inspected.
- The two-series chart palette was checked with a colorblind-safety validator
  (Okabe-Ito pair, worst-case protanopia Delta E 21.9).

### Not done by AI

- No experimental number, table entry, or figure in this repository was written by hand
  or estimated; every metric comes from an executed run whose artifacts are in
  `outputs/`.
- No dataset was fabricated or modified.
- The report PDF, slides, YouTube video, and the group's own conclusions are not covered
  by this entry. Add further entries if AI tools are used for those.

## Declaration

TODO - after review, complete: "The group has reviewed all AI-assisted code and can
explain it. Responsible member for Assignment 1 verification: <name>."
