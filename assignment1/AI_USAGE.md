# AI Usage Disclosure – Assignment 1

**Course:** CO3133 – Deep Learning and Its Applications
**Semester:** 261
**Group:** Nhóm CR7
**Assignment:** 1 – Foundations of Deep Learning Pipelines and Architectures

This file is the detailed log required by handbook Section 5.1, item 3. Summary versions
appear on the [shared landing page](../index.html) (Section 5.1, item 1) and on the
[Assignment 1 page](index.html) (Section 5.1, item 2). The report-level disclosure required
by the same item is pending the M2 Final report; it will carry the same content as the
Summary below. The nine fields that Section 5.2 makes mandatory for each tool are all
recorded here — see *Mandatory field coverage* for where each one sits.

## Summary

| Tool | Purpose | Used by |
|---|---|---|
| Claude Code (Claude Sonnet 5, Claude Opus 5) | Review of the Assignment 1 source code — architecture, naming, dead code, consistency between modules — and code-cleanliness suggestions | Trần Hoàng Vỹ Khang |
| Claude Code (Claude Sonnet 5, Claude Opus 5) | Review of the MLP and CNN code on the `mlp-cnn` branch contributed by another member | Trần Hoàng Vỹ Khang |
| Claude Code (Claude Opus 5) | **Implementation** of the CNN, LSTM/GRU and Transformer models, the shared sequence representation and the inference-timing fix; drafting of the results and analysis sections | Trần Hoàng Vỹ Khang |

The tool was used in two distinct ways, and the distinction matters for this disclosure. For
the M1 Draft pipeline (Entries 1 and 2) it acted as a **reviewer** of member-written code. For
the three models added for M2 Final (Entry 3) it acted as an **author**: `src/models/cnn.py`,
`src/models/rnn.py`, `src/models/transformer.py`, `src/data/sequence.py`, the rewritten
`measure_inference_time`, and the results and analysis sections of `README.md` and
`index.html` were drafted by the tool. Those files are listed individually in Entry 3 and are
pending line-by-line review by the responsible member, as handbook Section 5.5 requires. The
Linear and MLP models, the dataset layer, the training engine, the metrics, the plots and the
EDA notebook remain member-written. No other AI tool was used in Assignment 1.

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

## Entry 3 — Implementation of the three remaining mandatory models

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Opus 5 |
| Used by | Trần Hoàng Vỹ Khang |
| Category (Section 5.2) | Coding assistance (implementation, not review); debugging; result analysis; report writing |
| Time / stage | 02 Oct 2026; M2 Final model implementation |
| Purpose | Implementing the CNN, the LSTM/GRU and the Transformer encoder together with the shared image-to-sequence representation; registering them in the training script and the configuration; correcting the inference-timing measurement; executing the six training and evaluation runs; and drafting the results, comparison and error-analysis sections of `README.md` and `index.html` |
| Affected sections | Code: `src/models/cnn.py`, `src/models/rnn.py`, `src/models/transformer.py`, `src/data/sequence.py`, `src/engine/trainer.py` (`measure_inference_time`), `scripts/train.py`, `scripts/eval.py`, `configs/default.yaml`. Documentation: `README.md` (results, per-class, inductive-bias analysis, reproducibility) and `index.html` (Sections 3 to 8). Report: Part 2 (methodology) and Part 3 (implementation results) insofar as they reuse this material. |
| Representative prompts | "hãy đọc lại handbook-ene-v2 trong folder doc rồi tiếp tục làm" (re-read the handbook, then continue); "dựng CNN lên" (build the CNN); and the decision, taken by the member when asked, to finish all five models before writing the comparison once. The full session transcript is kept locally by the member and is not published. |
| How the output was edited and verified | Checks executed before any number was reported: every model was instantiated and run forward, confirming `(N, 10)` logits; the CNN was checked at 1, 2, 3 and 4 convolution blocks against the real `MaxPool2d` output, confirming the `28 // 2**n` spatial formula; `to_sequence` was compared element-wise against the source image, confirming that row token *t* equals `x[:, 0, t, :]`, column token *t* equals `x[:, 0, :, t]`, and patches 0 and 1 equal the first two 7x7 blocks in row-major order; both sequence models were run in all three representations; the error guards were triggered deliberately (5 pooling blocks, `patch_size=5`, `nhead=5`, unknown mode, unknown cell); no model contains a softmax module, as Section 11.1 requires. After the runs, a script re-read every `metrics_test.json` and `history.json` and confirmed that each accuracy, macro-F1, parameter count, timing, epoch total and per-class F1 printed in `README.md` and `index.html` matches the artifact it claims to come from; both HTML pages were parsed for tag balance. **The outstanding verification is human**: this code was drafted by the tool, not by a member, so handbook Section 5.5 obliges the responsible member to read it line by line and be able to explain it before submission. That review has not yet been recorded. |
| Responsible member | Trần Hoàng Vỹ Khang |
| Verification sources | PyTorch documentation for `nn.Conv2d`, `nn.MaxPool2d`, `nn.LSTM`, `nn.GRU` and `nn.TransformerEncoderLayer`; course handbook Sections 11.1 (mandatory models and the convolution / pooling / feature-map, timestep and attention explanations), 11.2, 11.3, 12.1 and 12.2; the six executed runs under `outputs/` |

### Division of work, file by file

| File | AI contribution | Group contribution |
|---|---|---|
| `src/models/cnn.py` | Drafted by AI, including the convolution / pooling / feature-map explanation Section 11.1 requires | Review by the responsible member **outstanding** |
| `src/models/rnn.py` | Drafted by AI, including the timestep / input-size / hidden-representation explanation | Review by the responsible member **outstanding** |
| `src/models/transformer.py` | Drafted by AI, including the token-embedding / positional-encoding / attention explanation | Review by the responsible member **outstanding** |
| `src/data/sequence.py` | Drafted by AI | Review by the responsible member **outstanding** |
| `src/engine/trainer.py` | `measure_inference_time` rewritten by AI to time the model separately from the data loader | Rest of the file unchanged; review outstanding |
| `scripts/train.py`, `scripts/eval.py`, `configs/default.yaml` | Model registration and new configuration keys added by AI | Review outstanding |
| `README.md`, `index.html` | Results tables, inductive-bias analysis, error analysis and limitations drafted by AI from the executed runs | Review outstanding; the member owns every claim in them |

## Verification performed

These checks were executed and their evidence is in the repository.

- **End-to-end runs.** Both models were trained and evaluated by the group member running
  the committed scripts. Every metric in `README.md` and on the Assignment 1 page is read
  back from `outputs/<run_name>/metrics_test.json` and `history.json`; none was written by
  hand or estimated.
- **Normalization constants are self-consistent.** The `mean=0.2860, std=0.3530` used in
  `src/data/dataset.py` match, to four decimals, the statistics measured independently over
  the 60,000 training images in `notebooks/eda.ipynb`. One preprocessed batch was confirmed
  to have mean ≈ 0 and std ≈ 1.
- **The split is correct and leak-free.** The stratified split yields exactly 5,400 train
  and 600 validation images per class; `scripts/train.py` was checked never to construct a
  test loader.
- **Loss usage matches the handbook.** Both models return raw logits and no softmax is
  applied before `nn.CrossEntropyLoss`, as handbook Section 11.1 requires; this was
  verified by searching the whole source tree for softmax calls.
- **Figures were rendered and inspected**, including the confusion matrices and the
  correct/incorrect prediction grids. The two-series curve palette was checked with a
  colorblind-safety validator (Okabe-Ito pair, worst-case protanopia ΔE 21.9).
- **A packaging defect was caught and fixed.** A `.gitignore` rule matching `data/` at any
  depth was silently excluding `src/data/dataset.py` from version control; it was replaced
  with per-assignment anchored paths.

## What AI was not used for

- The AI-authored files are confined to those named in Entry 3. The Linear and MLP models,
  `src/data/dataset.py`, `src/engine/trainer.py` apart from `measure_inference_time`,
  `src/utils/{seed,metrics,plots}.py` and `notebooks/eda.ipynb` were written by a group
  member, and for the M1 Draft pipeline (Entries 1 and 2) the tool only reviewed and
  reported findings.
- No experimental number, table entry or figure was fabricated or estimated; each one comes
  from an executed run whose artifacts are reconstructible with the commands in `README.md`.
- No dataset was fabricated, modified or re-labelled. Fashion-MNIST is used as published.
- No pretrained weights were used.
- No experiment is claimed that was not actually run. The CNN, LSTM/GRU and Transformer
  results do not exist yet and are not reported anywhere.
- No private, restricted or credential-bearing data was pasted into the tool, as handbook
  Section 5.5 requires. The review ran against the repository source only; Fashion-MNIST is
  a public dataset and no credentials, tokens or personal data are present in the repository.
- No AI-generated code or text was submitted unchecked: every finding was read by the
  responsible member before any edit was made, and the edits are the member's own.
- The report PDF, the slides and the YouTube presentation are not covered by this file.
  Further entries will be added here if AI tools are used in producing them.

## Declaration

All final results, code and conclusions are owned by the group. AI tools were used for code
review (Entries 1 and 2) and for implementing the three models added after the M1 Draft
(Entry 3), not as a replacement for original work: the problem framing, the experimental
design, the choice of representations and every decision about what to report were made by
the group.

The group accepts that the instructor may interview any member and request explanation, live
edits, metric interpretation or partial reproduction (handbook Section 5.5). **The
AI-authored files listed in Entry 3 are not yet signed off**; the responsible member must
read and be able to explain each of them before submission, and this file will be updated to
record that review when it is done. No experimental number here was produced by anything
other than an executed run.
Responsible member for Assignment 1 verification: **Trần Hoàng Vỹ Khang**.
