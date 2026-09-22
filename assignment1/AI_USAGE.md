# AI Usage Disclosure – Assignment 1

**Course:** CO3133 – Deep Learning and Its Applications  
**Semester:** 261  
**Group:** Nhóm CR7

## Tools Used

| Tool | Purpose | Who used it |
|------|---------|-------------|
| Claude Code (Claude Sonnet 5, Claude Opus 5) | Reviewing the MLP and CNN code on the `mlp-cnn` branch for cleanliness and consistency; the group re-checked the findings and decided not to merge that branch yet | Trần Hoàng Vỹ Khang |

## Details

### Claude Code — code cleanliness review

| Field | Value |
|---|---|
| Tool / model | Claude Code CLI; Claude Sonnet 5 and Claude Opus 5 |
| Used by | Trần Hoàng Vỹ Khang |
| Stage | Assignment 1 M1 Draft development; code review performed 22 September 2026 |
| Purpose | Checking the Assignment 1 source code for cleanliness: naming, structure, redundant or dead code, and consistency between modules |
| Affected sections | `assignment1/src/`, `assignment1/scripts/`, `assignment1/configs/` |
| Representative prompt | "review code của branch mlp-cnn" (review the code on the mlp-cnn branch); "phần AI usage đã bổ sung đúng theo requirement chưa" (is the AI usage disclosure complete per the requirement). The full session transcript is kept locally by the member and is not published. |
| How the output was edited and verified | The tool produced review findings only; no file on the `mlp-cnn` branch was modified as a result. Before the findings were reported, the `CNN` class was instantiated with 2, 3 and 4 convolution blocks and a forward pass was executed to confirm the output shape `(N, 10)` and that the `28 // 2**len(channels)` spatial-size formula matches the real `MaxPool2d` output. The responsible member read the findings and acted on them: the `mlp-cnn` branch was deliberately kept out of `a1-draft` in its current state, pending restoration of the deleted MLP docstring and the missing explanation of convolution, pooling and feature maps in `cnn.py`. |
| Responsible member | Trần Hoàng Vỹ Khang |
| Verification sources | Course handbook Section 11.1 (mandatory model requirements, including "Explain convolution, pooling, and feature maps") and Section 12.2 (fairness constraints); PyTorch documentation for `nn.Conv2d` and `nn.MaxPool2d` output shapes; the executed shape test described above. |

> **This disclosure is not complete yet.** Beyond the cleanliness review recorded above,
> AI assistance was also used earlier in this repository for environment setup, for the
> repository scaffolding, and for authoring parts of the Assignment 1 pipeline
> (`src/engine/trainer.py`, `src/utils/metrics.py`, `src/utils/plots.py`,
> `src/models/linear.py`, `src/models/mlp.py`, `scripts/train.py`, `scripts/eval.py`,
> `notebooks/eda.ipynb`, `README.md`), while `src/utils/seed.py` and the first version of
> `src/data/dataset.py` were written by a group member. A detailed log of that work was
> written earlier and is preserved in git history at commit `8615635`
> (`git show 8615635:assignment1/AI_USAGE.md`). Section 5.2 of the handbook requires
> those entries here as well; missing disclosure may be treated as an academic integrity
> violation. Merge the two versions before submitting.

## Declaration

All final results, code, and conclusions are reviewed, understood, and owned by the group. AI tools were used solely for assistance, not as a replacement for original work.
