# AI Usage Disclosure – Assignment 1

**Course:** CO3133 – Deep Learning and Its Applications
**Semester:** 261
**Group:** Nhóm CR7 (Trần Hoàng Vỹ Khang – `2352502`, Đỗ Đăng Khoa – `2352554`, Dương Đăng Khoa – `2352551`)
**Assignment:** 1 – Foundations of Deep Learning Pipelines and Architectures

This file is the detailed log required by handbook Section 5.1, item 3. Summary versions
appear on the [shared landing page](../index.html) (Section 5.1, item 1), on the
[Assignment 1 page](index.html) (Section 5.1, item 2), and in the M2 Final Report. The nine
fields that Section 5.2 makes mandatory for each tool are all recorded here — see
*Mandatory field coverage* for where each one sits.

## Summary (All Group Members)

| Tool | Purpose | Used by / Verified by |
|---|---|---|
| Claude Code (Claude Sonnet 5, Claude Opus 5) | Review of the Assignment 1 M1 Draft source code — architecture, naming, dead code, consistency between modules — and code-cleanliness suggestions (Entry 1) | Used by: Trần Hoàng Vỹ Khang (`2352502`); Linear/MLP baselines co-verified by Đỗ Đăng Khoa (`2352554`) |
| Claude Code (Claude Sonnet 5, Claude Opus 5) | Architecture and cleanliness review of the MLP and CNN code on the `mlp-cnn` branch contributed by Đỗ Đăng Khoa (Entry 2) | Used by: Trần Hoàng Vỹ Khang (`2352502`); branch authored by Đỗ Đăng Khoa (`2352554`) |
| Claude Code (Claude Opus 5) | Architecture control and code review of the M2 Final models (`cnn.py`, `rnn.py`, `transformer.py`, `sequence.py`), inference-timing isolation, and cross-checking of reported tables against run JSON artifacts (Entry 3) | Used by: Trần Hoàng Vỹ Khang (`2352502`); results & sequence models co-verified by Đỗ Đăng Khoa (`2352554`) and Dương Đăng Khoa (`2352551`) |
| Google Antigravity (AGY) Agent | Code review of combined 6-model curve/confusion-matrix plotting in `scripts/compare.py` and scientific audit of the M2 Final English report so claims match single-seed evidence (Entry 4) | Used & verified by: Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), Dương Đăng Khoa (`2352551`) |

Across all four entries, the tools acted strictly as **code reviewers, architecture controllers and scientific auditors**, not as direct code modifiers or autonomous authors. For every module — the M1 Draft pipeline (Entries 1 and 2), the CNN, LSTM/GRU, Transformer, sequence representation and timing instrumentation added for M2 Final (Entry 3), and the combined multi-model plotting and report verification (Entry 4) — the tools inspected member-written files, verified tensor shapes, inductive-bias constraints and handbook compliance, and flagged findings for the responsible members to resolve. All source files, model implementations, training runs and report analyses were written, edited and verified line by line by all three group members (Trần Hoàng Vỹ Khang, Đỗ Đăng Khoa, Dương Đăng Khoa), as handbook Section 5.5 requires.

## Mandatory field coverage (handbook Section 5.2)

| Required field | Where it is recorded |
|---|---|
| Tool name and version/model | *Tool / model* row of each entry |
| Member who used it | *Used by* row of each entry |
| Time or development stage | *Time / stage* row of each entry |
| Purpose | *Purpose* and *Category* rows of each entry |
| Affected assignment section(s) | *Affected sections* row of each entry |
| Representative prompt example or prompt-log link | *Representative prompts* row of each entry |
| How AI output was edited and verified | *How the output was edited and verified* row, plus *Review findings acted on* and *Verification performed* |
| Member responsible for final verification | *Responsible member* row of each entry |
| Sources used for verification | *Verification sources* row of each entry |

## Entry 1 — Review and cleanup of the training pipeline

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Sonnet 5 and Claude Opus 5 |
| Used by | Trần Hoàng Vỹ Khang |
| Time / stage | 16 Sep 2026; M1 Draft consolidation |
| Category (Section 5.2) | Coding assistance (review only); debugging; test generation |
| Purpose | Reviewing the member-written pipeline for correctness and cleanliness: syntax and indentation errors, missing preprocessing steps, module structure and naming, redundant or dead code, consistency between the data, model, engine and script layers, and repository hygiene |
| Affected sections | Code: `assignment1/src/`, `assignment1/scripts/`, `assignment1/configs/`, `assignment1/notebooks/eda.ipynb`, `assignment1/README.md`, root `.gitignore`. Report: Part 1 (data splitting and preprocessing) and Part 3 (implementation results), insofar as they describe code this review touched. Assignment page: Dataset description and EDA, Experimental setup, Results. Code changes arising from the review were made by the responsible member. |
| Representative prompts | "review code và làm sạch code" (review and clean up the code); "kiểm tra lại phần chuẩn hoá và chia tập dữ liệu" (re-check the normalization and the data split); "phần AI usage đã bổ sung đúng theo requirement chưa" (is the AI usage disclosure complete per the requirement). The full session transcript is kept locally by the member and is not published. |
| How the output was edited and verified | The tool produced findings; the responsible member read each one, decided whether to act on it, and made the edits. Findings were checked against the code and the handbook before being accepted — see *Review findings acted on* and *Verification performed* below. Two findings were material: a `.gitignore` rule that silently excluded `src/data/dataset.py` from version control, and a syntax error plus a wrong indentation level in `src/utils/seed.py`. |
| Responsible member | Trần Hoàng Vỹ Khang |
| Verification sources | PyTorch and torchvision documentation (`nn.CrossEntropyLoss`, `datasets.FashionMNIST`, `transforms`); scikit-learn documentation (`f1_score`, `confusion_matrix`); course handbook Sections 10, 11.1, 12.1 and 12.2; the executed runs and the measured statistics in `notebooks/eda.ipynb` |

### Review findings acted on, file by file

Every file below was **written by the group member**. The table records what the review
flagged and who resolved it.

| File | What the review flagged | Resolved by |
|---|---|---|
| `src/utils/seed.py` | A syntax error and a wrong indentation level | Member |
| `src/data/dataset.py` | Normalization missing from the transform chain; the split was not stratified; no split statistics were reported back to the caller | Member (added normalization, stratified splitting and the `info` dictionary) |
| `src/utils/metrics.py` | Naming and return-type consistency with `scripts/eval.py` | Member |
| `src/utils/plots.py` | The two-series curve palette was not colorblind-safe | Member (switched to an Okabe-Ito pair) |
| `src/models/linear.py` | No finding beyond confirming raw logits are returned | — |
| `src/models/mlp.py` | No finding beyond confirming raw logits are returned | — |
| `src/engine/trainer.py` | Train/validation loop structure and the absence of any test-set access during training | — (confirmed correct) |
| `scripts/train.py` | Confirmed no test loader is constructed | — (confirmed correct) |
| `scripts/eval.py` | Confirmed metrics are read back from artifacts rather than recomputed inconsistently | — (confirmed correct) |
| `configs/default.yaml` | Key names consistent with the arguments the scripts read | — (confirmed correct) |
| `notebooks/eda.ipynb` | Measured statistics cross-checked against the constants hard-coded in `dataset.py` | — (confirmed consistent) |
| `README.md`, `index.html` | Wording and the reproducibility commands; every reported number traced back to an artifact file | Member |
| root `.gitignore` | A `data/` rule matching at any depth silently excluded `src/data/dataset.py` from version control | Member (replaced with per-assignment anchored paths) |

## Entry 2 — Code architecture and cleanliness review, `mlp-cnn` branch

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Sonnet 5 and Claude Opus 5 |
| Used by | Trần Hoàng Vỹ Khang |
| Time / stage | 22 Sep 2026; M1 Draft consolidation |
| Category (Section 5.2) | Coding assistance (review only); architecture review; test generation |
| Purpose | Checking the MLP and CNN code on the `mlp-cnn` branch, contributed by another member, for architecture and cleanliness: naming, module structure, redundant or dead code, and consistency with the modules already on `a1-draft` |
| Affected sections | Code: `assignment1/src/models/`, and the changes the branch makes under `assignment1/src/` and `assignment1/configs/`. Report and assignment page: none — the branch is not merged, so no result or claim from it is reported anywhere. |
| Representative prompts | "review code của branch mlp-cnn" (review the code on the `mlp-cnn` branch) |
| How the output was edited and verified | The tool produced findings only; no file on the `mlp-cnn` branch was modified as a result. Before the findings were reported, the `CNN` class was instantiated with 2, 3 and 4 convolution blocks and a forward pass was executed, confirming the output shape `(N, 10)` and that the `28 // 2**len(channels)` spatial-size formula matches the real `MaxPool2d` output. The responsible member read the findings and acted on them: the `mlp-cnn` branch was deliberately kept out of `a1-draft` and is not part of the M1 Draft submission. The group then decided not to repair that branch at all, so no MLP or CNN result from it is reported anywhere. The CNN that handbook Section 11.1 lists as a mandatory model, together with the explanation of convolution, pooling and feature maps that the same section requires, will be implemented fresh for M2 Final. |
| Responsible member | Trần Hoàng Vỹ Khang |
| Verification sources | Course handbook Sections 11.1 and 12.2; PyTorch documentation for `nn.Conv2d` and `nn.MaxPool2d` output shapes; the executed shape test described above |

## Entry 3 — Architecture control and code review of the M2 Final models and evaluation

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Opus 5 |
| Used by | Trần Hoàng Vỹ Khang |
| Category (Section 5.2) | Architecture review; code review (no direct code modification); debugging; result verification |
| Time / stage | 02–05 Oct 2026; M2 Final model review and deterministic verification |
| Purpose | Reviewing the architecture and implementation of the CNN, LSTM/GRU and Transformer encoder together with the shared image-to-sequence representation; checking model registration in the training script and configuration; reviewing inference-timing isolation (`measure_inference_time`) and cuDNN determinism settings; and cross-checking the comparison script (`scripts/compare.py`) and reported tables in `README.md` and `index.html` against the JSON run artifacts |
| Affected sections | Code reviewed: `src/models/cnn.py`, `src/models/rnn.py`, `src/models/transformer.py`, `src/data/sequence.py`, `src/engine/trainer.py` (`measure_inference_time`), `src/utils/seed.py`, `scripts/train.py`, `scripts/eval.py`, `scripts/compare.py`, `configs/default.yaml`. Documentation reviewed: `README.md` (results, per-class, inductive-bias analysis, reproducibility) and `index.html` (Sections 3 to 8). Report: Part 2 (methodology) and Part 3 (implementation results) insofar as they describe these modules. |
| Representative prompts | "hãy đọc lại handbook-ene-v2 trong folder doc rồi kiểm tra kiến trúc các model M2" (re-read the handbook, then check the M2 model architectures); "review file cnn.py, rnn.py, transformer.py và kiểm tra tính nhất quán số liệu" (review cnn.py, rnn.py, transformer.py and check metric consistency). The full session transcript is kept locally by the member and is not published. |
| How the output was edited and verified | The tool produced architectural and code-review findings only; all code edits and training runs were performed by the group members. Checks executed during verification: every model was instantiated and run forward, confirming `(N, 10)` logits; the CNN was checked at 1, 2, 3 and 4 convolution blocks against the real `MaxPool2d` output, confirming the `28 // 2**n` spatial formula; `to_sequence` was compared element-wise against the source image, confirming that row token *t* equals `x[:, 0, t, :]`, column token *t* equals `x[:, 0, :, t]`, and patches 0 and 1 equal the first two 7x7 blocks in row-major order; both sequence models were run in all three representations; the error guards were triggered deliberately (5 pooling blocks, `patch_size=5`, `nhead=5`, unknown mode, unknown cell); no model contains a softmax module, as Section 11.1 requires. After the runs, `scripts/compare.py` re-read every `metrics_test.json` and `history.json` and confirmed that each accuracy, macro-F1, parameter count, timing, epoch total and per-class F1 printed in `README.md` and `index.html` matches the artifact it comes from; both HTML pages were parsed for tag balance. |
| Responsible member | Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`) |
| Verification sources | PyTorch documentation for `nn.Conv2d`, `nn.MaxPool2d`, `nn.LSTM`, `nn.GRU` and `nn.TransformerEncoderLayer`; course handbook Sections 11.1 (mandatory models and the convolution / pooling / feature-map, timestep and attention explanations), 11.2, 11.3, 12.1 and 12.2; the six executed runs under `outputs/` |

### Review findings acted on, file by file

| File | What the architecture / code review checked | Resolved & verified by |
|---|---|---|
| `src/models/cnn.py` | Spatial dimension reduction (`28 // 2**n`), conv-bn-relu-pool block structure, absence of pre-loss softmax, and completeness of the Section 11.1 feature-map explanation | Member |
| `src/models/rnn.py` | Timestep / input-size / hidden-state dimensions across row, column and patch modes, final-step vs mean-pool readout | Member |
| `src/models/transformer.py` | Linear token projection to `d_model`, learnable positional encoding shape, `batch_first=True` encoder layers, and divisibility guard for `d_model % nhead` | Member |
| `src/data/sequence.py` | Exact element-wise correspondence of row, column and non-overlapping patch sequences to the `(N, 1, 28, 28)` input tensor | Member |
| `src/engine/trainer.py`, `src/utils/seed.py` | Flagged that end-to-end timing included DataLoader overhead (leading to separate model-only timing with CUDA synchronization) and that CNN reproducibility required `cudnn.deterministic=True` | Member |
| `scripts/train.py`, `scripts/eval.py`, `scripts/compare.py`, `configs/default.yaml` | Consistency of CLI model choices, config hyperparameters, and automated aggregation of the four comparison tables from JSON artifacts | Member |
| `README.md`, `index.html` | Cross-checked all reported metrics, per-class F1 scores, confusion counts and inductive-bias claims against `outputs/*/metrics_test.json` and `history.json` | Member |

## Entry 4 — Review of combined multi-model figures (`compare.py`) and scientific audit of M2 Final report

| Field | Value |
|---|---|
| Tool / model | Google Antigravity (AGY) Agent |
| Used by | Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`) |
| Category (Section 5.2) | Code review (plotting layout); scientific & numerical consistency audit (no direct code modification) |
| Time / stage | Oct 2026; M2 Final report verification and cross-artifact reconciliation |
| Purpose | Reviewing `scripts/compare.py` for combined 6-model learning curve (`assets/curves_all_models.png`) and confusion matrix (`assets/confusion_matrix_all_models.png`) figure generation; auditing `CR7_A1_Report.md`, `README.md` and `index.html` so every number, caption, error count, analytical FLOPs/MACs figure and scientific claim matches the single-seed (`seed = 42`) JSON artifacts |
| Affected sections | Code & Figures reviewed: `scripts/compare.py`, `assets/curves_all_models.png`, `assets/confusion_matrix_all_models.png`. Documentation & Report reviewed: `CR7_A1_Report.md`, `README.md`, `index.html`. |
| Representative prompts | "kiểm tra lại số liệu giữa README, index.html và báo cáo CR7_A1_Report.md xem đã khớp hoàn toàn với JSON trong outputs/ chưa"; "rà soát phần phân tích lỗi, FLOPs/MACs và kết luận theo đúng tiêu chí Section 12-13 của handbook". |
| How the output was edited and verified | Cross-checked every cell of Tables 1–4 directly against `outputs/*_fashion_mnist_seed42/{history,metrics_test,config}.json` (confirming e.g. CNN `cm[6][2]=33 + cm[6][4]=64 = 97`, GRU vs. LSTM epoch-by-epoch train/val loss across all 15 epochs, and analytical MACs/FLOPs per image); group members applied all edits and verified the rendered figures. |
| Responsible member | Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`) |
| Verification sources | Committed JSON artifacts in `assignment1/outputs/`, `src/data/dataset.py`, `notebooks/eda.ipynb`, and Course Handbook Sections 3.1–3.3, 5.2, and 10–13 |

## Verification performed

These checks were executed and their evidence is in the repository.

- **End-to-end runs.** All six models (`linear`, `mlp`, `cnn`, `lstm`, `gru`, `transformer`)
  were trained and evaluated on the group hardware running the committed scripts. Every
  metric in `README.md` and on the Assignment 1 page is read back from
  `outputs/<run_name>/metrics_test.json` and `history.json` via `scripts/compare.py`; none
  was written by hand or estimated.
- **Normalization constants are self-consistent.** The `mean=0.2860, std=0.3530` used in
  `src/data/dataset.py` match, to four decimals, the statistics measured independently over
  the 60,000 training images in `notebooks/eda.ipynb`. One preprocessed batch was confirmed
  to have mean ≈ 0 and std ≈ 1.
- **The split is correct and leak-free.** The stratified split yields exactly 5,400 train
  and 600 validation images per class; `scripts/train.py` was checked never to construct a
  test loader.
- **Loss usage matches the handbook.** All models return raw logits and no softmax is
  applied before `nn.CrossEntropyLoss`, as handbook Section 11.1 requires; this was
  verified by searching the whole source tree for softmax calls.
- **Figures were rendered and inspected**, including the confusion matrices and the
  correct/incorrect prediction grids. The two-series curve palette was checked with a
  colorblind-safety validator (Okabe-Ito pair, worst-case protanopia ΔE 21.9).
- **A packaging defect was caught and fixed.** A `.gitignore` rule matching `data/` at any
  depth was silently excluding `src/data/dataset.py` from version control; it was replaced
  with per-assignment anchored paths.

## What AI was not used for

- All source files in `assignment1/` — the Linear, MLP, CNN, LSTM/GRU and Transformer models,
  `src/data/{dataset,sequence}.py`, `src/engine/trainer.py`, `src/utils/{seed,metrics,plots}.py`,
  `scripts/{train,eval,compare}.py` and `notebooks/eda.ipynb` — were written and edited by the
  group members. AI tools were used strictly to review files, control architectural
  constraints, and verify consistency across code, artifacts and documentation.
- No experimental number, table entry or figure was fabricated or estimated; each one comes
  from an executed run whose artifacts are reconstructible with the commands in `README.md`.
- No dataset was fabricated, modified or re-labelled. Fashion-MNIST is used as published.
- No pretrained weights were used.
- No experiment is claimed that was not actually run. All six reported models have their
  committed `config.json`, `history.json` and `metrics_test.json` under `outputs/`.
- No private, restricted or credential-bearing data was pasted into the tool, as handbook
  Section 5.5 requires. The review ran against the repository source only; Fashion-MNIST is
  a public dataset and no credentials, tokens or personal data are present in the repository.
- Every review finding was inspected and verified by the responsible members before any edit
  was made by the group.

## Declaration

All final results, code and conclusions are written, verified and owned by the group. AI
tools were used strictly for code review, architecture control and artifact consistency
verification (Entries 1–4), not for direct code modification or as a replacement for
original work: the problem framing, the model implementations, the experimental design, the
choice of representations and every decision about what to report were made by the group.

The group accepts that the instructor may interview any member and request explanation, live
edits, metric interpretation or partial reproduction (handbook Section 5.5). Every file in
this repository has been written and verified line by line by the responsible members. No
experimental number here was produced by anything other than an executed run.
Responsible members for Assignment 1 verification: **Trần Hoàng Vỹ Khang**, **Đỗ Đăng Khoa**, **Dương Đăng Khoa**.
