# Assignment 1 - Foundations of Deep Learning Pipelines and Architectures

From Linear Models to Modern Sequence Models: A Comparative Study for Image
Classification.

- **Course:** CO3133 Deep Learning and Its Applications, Semester-261
- **Instructor:** Lê Thành Sách
- **Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM - Faculty of Computer Science and Engineering
- **Dataset:** Fashion-MNIST (main results); MNIST for debugging only

## Milestone status

| Milestone | Due | Required minimum | State |
|---|---|---|---|
| M1 Draft (25% of A1) | 23 Sep 2026 | EDA, Dataset/DataLoader, train/val loop, Linear + MLP runnable | **complete**, submitted at commit `8476977` (22 Sep 2026) |
| M2 Final (75% of A1) | 21 Oct 2026 | all five models, full comparison, report, slides, video, page, checkpoints | **models and comparison complete**; report, slides and video outstanding |

All five mandatory models of handbook Section 11.1 are now implemented, trained and
compared: Linear/softmax, MLP, CNN, LSTM **and** GRU (training both is also the optional
Section 11.2 extension), and Transformer. What remains for M2 Final is the report PDF, the
slides, the YouTube video, and `scripts/compare.py`, which is still an empty stub - the
tables below were assembled directly from the `metrics_test.json` of each run.

## Project layout

```
assignment1/
├── configs/default.yaml        experiment configuration (seed, data, model, training)
├── src/
│   ├── data/dataset.py         Fashion-MNIST/MNIST loading, stratified split, DataLoaders
│   ├── data/sequence.py        image -> row / column / patch token sequence
│   ├── models/linear.py        linear / softmax classifier
│   ├── models/mlp.py           multilayer perceptron
│   ├── models/cnn.py           self-designed conv-BN-ReLU-pool CNN
│   ├── models/rnn.py           LSTM / GRU over a token sequence
│   ├── models/transformer.py   Transformer encoder, CLS token + learned positions
│   ├── engine/trainer.py       train / validate loops, checkpointing, timing
│   └── utils/{seed,metrics,plots}.py     reproducibility, accuracy + macro-F1, figures
├── scripts/train.py            train one model, write checkpoint + curves
├── scripts/eval.py             evaluate a checkpoint on the test split
├── scripts/compare.py          gather metrics of all five models (M2, empty)
├── notebooks/eda.ipynb         exploratory data analysis (executed, outputs included)
└── outputs/                    run artifacts (git-ignored)
```

## Installation

The virtual environment lives at the repository root, one level above this folder.

```powershell
# from the repository root
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r assignment1\requirements.txt
```

`requirements.txt` pins the CUDA 13.0 builds of torch/torchvision that produced the
reported numbers. On a machine without an NVIDIA GPU, remove the two `+cu130` pins and
the `--extra-index-url` line; everything runs on CPU unchanged, only slower.

Check the GPU is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.14.0+cu130 True
```

## Dataset preparation

No manual download. `torchvision` fetches Fashion-MNIST into `assignment1/data/` on
first use (~30 MB, git-ignored):

```powershell
python -c "from torchvision import datasets; datasets.FashionMNIST('assignment1/data', download=True)"
```

The scripts resolve `data_root` against the project root, so they can be launched from
any working directory.

## Train

```powershell
python assignment1\scripts\train.py --model linear
python assignment1\scripts\train.py --model mlp
python assignment1\scripts\train.py --model cnn
python assignment1\scripts\train.py --model lstm
python assignment1\scripts\train.py --model gru
python assignment1\scripts\train.py --model transformer
```

Every setting comes from `configs/default.yaml`; the flags `--model`, `--dataset`,
`--epochs`, `--batch-size`, `--lr`, `--seed`, `--run-name` override it for one run. Each
run writes `outputs/<run_name>/` containing `best.pt`, `history.json`, `curves.png` and
`config.json` (resolved config + split statistics + parameter count + hardware and
library versions).

`scripts/train.py` never reads the test split.

## Evaluate

```powershell
python assignment1\scripts\eval.py --run-dir assignment1\outputs\linear_fashion_mnist_seed42
python assignment1\scripts\eval.py --run-dir assignment1\outputs\mlp_fashion_mnist_seed42
```

Loads the selected checkpoint and writes `metrics_test.json` (accuracy, macro-F1,
per-class F1, parameter count, both inference timings, confusion matrix),
`confusion_matrix_test.png`, and correct/incorrect prediction examples into the same run
directory. Use `--split val` to evaluate on validation instead.

## Exploratory data analysis

```powershell
jupyter notebook assignment1\notebooks\eda.ipynb   # or open it in VS Code
```

Select the `.venv` interpreter as the kernel. The committed notebook already contains its
outputs. It covers class distribution, input size, imbalance analysis (imbalance ratio and
normalized entropy), representative samples per class, per-class mean images, pixel
statistics, and one example batch after preprocessing.

Key facts it establishes:

| Item | Value |
|---|---|
| Input | 1x28x28 grayscale, 784 features flattened |
| Classes | 10, exactly balanced (imbalance ratio 1.0000, normalized entropy 1.0000) |
| Split | 54,000 train / 6,000 validation (stratified, seed 42) / 10,000 official test |
| Preprocessing | `ToTensor`, `Normalize(mean=0.2860, std=0.3530)`; no augmentation in the draft |
| Normalization check | measured mean/std over the 60,000 training images = 0.2860 / 0.3530 |
| Trivial baseline | 10% accuracy |

## Results (Fashion-MNIST, seed 42)

Held-out official test split, 10,000 images. Every model shares the identical stratified
54,000 / 6,000 split, the same Adam setting, the same 15-epoch budget with early stopping at
patience 5, and the same checkpoint rule - best validation macro-F1 - as handbook Section
12.2 requires.

| Model | Params | Test accuracy | Test macro-F1 | Best epoch | Train time | Inference, model only | Inference, end to end |
|---|---|---|---|---|---|---|---|
| Linear / softmax | 7,850 | 0.8440 | 0.8441 | 11 / 15 | 325.0 s (21.7 s/epoch) | 0.0005 ms/image | 0.2151 ms/image |
| MLP (784-256-10, dropout 0.2) | 203,530 | 0.8909 | 0.8902 | 15 / 15 | 301.6 s (20.1 s/epoch) | 0.0016 ms/image | 0.2179 ms/image |
| **CNN (32-64 conv-pool, dropout 0.2)** | **50,378** | **0.9154** | **0.9145** | 9 / 14 | 216.8 s (15.5 s/epoch) | 0.0097 ms/image | 0.2368 ms/image |
| LSTM (28 row steps, hidden 128) | 82,186 | 0.8906 | 0.8898 | 13 / 15 | 168.9 s (11.3 s/epoch) | 0.0086 ms/image | 0.1615 ms/image |
| GRU (28 row steps, hidden 128) | 61,962 | 0.9017 | 0.9010 | 13 / 15 | 220.5 s (14.7 s/epoch) | 0.0051 ms/image | 0.1587 ms/image |
| Transformer (28 row tokens, d_model 128, 4 heads, 2 layers) | 274,058 | 0.8846 | 0.8825 | 15 / 15 | 250.4 s (16.7 s/epoch) | 0.0152 ms/image | 0.2302 ms/image |

Ranking by test macro-F1: **CNN 0.9145 > GRU 0.9010 > MLP 0.8902 ~ LSTM 0.8898 >
Transformer 0.8825 > Linear 0.8441.** Accuracy and macro-F1 agree to within 0.003 for every
model, which is what the exactly balanced class distribution predicts.

### Per-class F1 on the test split

Sorted by how hard the class is for the linear baseline.

| Class | Linear | MLP | CNN | LSTM | GRU | Transformer |
|---|---|---|---|---|---|---|
| Shirt | 0.5999 | 0.7138 | **0.7399** | 0.7024 | 0.7220 | 0.6636 |
| Pullover | 0.7323 | 0.8091 | **0.8699** | 0.8235 | 0.8431 | 0.8129 |
| Coat | 0.7442 | 0.8236 | **0.8737** | 0.8229 | 0.8419 | 0.8094 |
| T-shirt/top | 0.8073 | 0.8430 | **0.8629** | 0.8326 | 0.8503 | 0.8343 |
| Dress | 0.8492 | 0.8870 | **0.9094** | 0.8905 | 0.9004 | 0.8777 |
| Sneaker | 0.9238 | 0.9509 | **0.9652** | 0.9439 | 0.9575 | 0.9491 |
| Sandal | 0.9314 | 0.9641 | **0.9832** | 0.9619 | 0.9759 | 0.9670 |
| Bag | 0.9355 | 0.9716 | **0.9798** | 0.9790 | 0.9755 | 0.9692 |
| Ankle boot | 0.9499 | 0.9594 | **0.9728** | 0.9599 | 0.9630 | 0.9630 |
| Trouser | 0.9672 | 0.9800 | **0.9885** | 0.9814 | 0.9804 | 0.9784 |

The CNN is the best model on all ten classes, so the ranking above is not an artifact of
averaging.

### How data representation and inductive bias affect performance

Splitting the ten classes into the **four upper-body garments** that share a silhouette
(Shirt, Pullover, Coat, T-shirt/top) and the **other six**, which all have a distinctive
global shape, makes the pattern explicit:

| Model | Mean F1, upper-body 4 | Mean F1, other 6 | Gap |
|---|---|---|---|
| Linear / softmax | 0.7209 | 0.9261 | 0.2052 |
| MLP | 0.7974 | 0.9522 | 0.1548 |
| **CNN** | **0.8366** | **0.9664** | **0.1298** |
| LSTM | 0.7954 | 0.9528 | 0.1574 |
| GRU | 0.8143 | 0.9588 | 0.1444 |
| Transformer | 0.7801 | 0.9507 | 0.1707 |

- **Inductive bias beats raw capacity.** The CNN wins with **50,378 parameters - 4.0x fewer
  than the MLP and 5.4x fewer than the Transformer** - for +0.024 macro-F1 over the MLP.
  Convolution supplies two priors for free that the other models must learn from 54,000
  images: neighbouring pixels are related, and a feature means the same thing wherever it
  appears. On a dataset this size that is worth more than parameters.
- **The whole ranking is decided inside the upper-body cluster.** The other six classes are
  effectively saturated - every model except the linear one is above 0.94 on them, and the
  spread across models is 0.016. The upper-body spread is 0.057, 3.5x wider. Those four
  classes differ in local texture (a knit surface, a button placket, a collar) at exactly
  the scale a 3x3 kernel over a 28x28 image resolves, which is why the model gap and the
  cluster gap close together.
- **Shirt is the floor for every architecture**, 0.5999 to 0.7399. The top confusions of the
  CNN are Shirt into T-shirt/top (146 images), Shirt into Pullover (75) and Shirt into Coat
  (75). Even the best model leaves Shirt 0.23 F1 below its own average, so this is label
  ambiguity in Fashion-MNIST at 28x28 grayscale, not a capacity limit - no model here fixes
  it and scaling one up is unlikely to.
- **The sequence models land between the MLP and the CNN, and the row representation says
  why.** Reading the image as 28 row tokens destroys the vertical axis but keeps each row
  intact, so a GRU retains a weaker spatial prior than convolution and a stronger one than
  the MLP, which flattens both axes away. The result follows that ordering exactly: GRU
  0.9010 > MLP 0.8902, and with **3.3x fewer parameters** (61,962 against 203,530).
- **GRU beats LSTM here: +0.011 macro-F1 with 75% of the parameters.** Over a 28-step
  sequence the separate cell state of the LSTM has no long-range dependency to protect, so
  its third gate only adds parameters to overfit with. This is the optional Section 11.2
  comparison, and on this task the simpler cell is the better one.
- **The Transformer is the weakest non-linear model despite having the most parameters.**
  It starts with no locality prior at all - self-attention treats the 28 rows as an unordered
  set and has to learn adjacency through the learned positional encoding - and its
  upper-body gap (0.1707) is the widest of any non-linear model. Its error profile shows the
  same thing in detail: Shirt into Pullover 113 confusions against 75 for the CNN, and Coat
  into Pullover 98 against 45. The gap between CNN and Transformer is not spread across the
  dataset; it is almost entirely these texture classes.

### What this comparison does not show

- **Two models are not converged.** The MLP and the Transformer both selected the last epoch
  in the budget (15 / 15) with validation macro-F1 still rising, so 0.8902 and 0.8825 are
  lower bounds for those architectures, not their ceilings. The Transformer is the one most
  penalised by a fixed 15-epoch budget, since attention models trained from scratch
  typically need the longest schedule. The CNN, by contrast, peaked at epoch 9 and
  early-stopped at 14. **A conclusion of the form "CNN beats Transformer on Fashion-MNIST"
  is not supported by this table**; what is supported is "CNN beats Transformer under an
  equal and short epoch budget".
- **No parameter-matched or compute-matched claim.** Section 11.2 lists a matched-budget
  comparison as an optional extension and it was not run. The models differ by 35x in
  parameter count, so the table compares six specific configurations, not six architecture
  families.
- **The training-time column is not a model property.** With `num_workers: 0` the pipeline is
  bound by single-process CPU preprocessing, so the per-epoch figures mostly measure the data
  loader. The ordering is not even monotone in model cost - the LSTM run averaged
  11.3 s/epoch against 21.7 for the linear model - and run-to-run machine noise exceeds the
  real difference: during the MLP run the EDA notebook was executing on the same machine,
  which is why its epochs 5-7 are visibly faster in `history.json`. Do not rank models by
  this column.
- **The two inference columns answer different questions.** `ms_per_image` times the model
  alone, with batches already on the GPU, and is monotone in computation as it should be:
  linear 0.0005 < MLP 0.0016 < GRU 0.0051 < LSTM 0.0086 < CNN 0.0097 < Transformer 0.0152.
  The end-to-end column adds the loader and collapses to 0.16-0.24 ms/image for every model,
  because loading costs **15x to 400x** the forward pass at this scale. Deployment throughput
  here is a data-pipeline property; only the model-only column compares architectures.

## Reproducibility

| Item | Value |
|---|---|
| Seed | 42, applied to `random`, `numpy`, `torch`, `torch.cuda` (`src/utils/seed.py`) |
| Split | stratified 90/10 of the official training set; official test set untouched during training |
| Split unit | one image (no grouping structure in Fashion-MNIST) |
| Checkpoint rule | best validation macro-F1; early stopping after 5 epochs without gain |
| Optimizer | Adam, lr 1e-3, weight decay 0, batch size 128, max 15 epochs - identical for all six runs |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11 |
| Software | Python 3.13.14, torch 2.14.0+cu130, torchvision 0.29.0+cu130, numpy 2.5.3, scikit-learn 1.9.1 |
| Mixed precision | not used |
| Code revision, Linear + MLP | tag `a1-draft`, commit `8476977` |
| Code revision, CNN + LSTM + GRU + Transformer | **uncommitted working tree, 02 Oct 2026** - see the warning below |

Every number reported above is traceable to a run directory under `outputs/`: the model
configuration and library versions are in its `config.json`, the per-epoch curve data in
`history.json`, the test metrics in `metrics_test.json`, and the weights in `best.pt`. The
`run_name` printed in each of those files is the experiment ID.

**The five-model table is not yet reproducible from a commit.** The Linear and MLP runs were
performed at commit `8615635` and the code is unchanged through tag `a1-draft` (commit
`8476977`): `git diff 8615635 tags/a1-draft -- src scripts configs` is empty, so either
commit reproduces those two rows. The CNN, LSTM, GRU and Transformer runs were produced by a
working tree that adds `src/models/{cnn,rnn,transformer}.py`, `src/data/sequence.py` and a
rewritten `measure_inference_time`, and **that tree has not been committed**. Handbook
Section 4.2 requires every main result to be traceable to a commit or tag, so commit this
tree and record its hash here before the report or the assignment page cites these numbers.

Two further notes on revision hygiene:

- The re-measured inference timings supersede the ones in the M1 Draft page. The Linear and
  MLP checkpoints were not retrained; only `scripts/eval.py` was re-run against them, so
  their accuracy and macro-F1 are unchanged from the M1 submission while their inference
  columns are new.
- Write `tags/a1-draft`, not `a1-draft`, in any command that has to mean the M1 tag. A branch
  of the same name exists, so the bare name is ambiguous once that branch moves past the tag.

**Why there are two inference columns.** With `num_workers: 0` the pipeline is bound by
single-process CPU preprocessing, so timing a loop over the DataLoader measures the loader
and not the model - that is how the M1 Draft page came to show a CNN as faster than a single
linear layer. `measure_inference_time` now moves every batch to the device before the clock
starts and reports that as `ms_per_image`, keeping the loader-inclusive figure separately as
`end_to_end_ms_per_image`. The training-time column is still loader-bound and is documented
as not comparable across models.

## Deliverables

| Deliverable | Where | State |
|---|---|---|
| Source code | this directory | all five mandatory models implemented |
| Assignment 1 page | [index.html](index.html) - published at <https://thvkhang.github.io/ALearning/assignment1/> | five-model results |
| Shared landing page | [../index.html](../index.html) - published at <https://thvkhang.github.io/ALearning/> | complete |
| AI Usage Disclosure | [AI_USAGE.md](AI_USAGE.md), plus a summary on both pages | complete; report-level disclosure pending M2 |
| Checkpoints | `outputs/<run_name>/best.pt`, reconstructed with the commands above | reproducible from source |
| Full comparison script | `scripts/compare.py` | **empty stub**; tables assembled by hand from `metrics_test.json` |
| Report PDF (`CR7_A1_Report.pdf` on the LMS) | - | M2 Final |
| Slides | - | M2 Final |
| YouTube video | - | M2 Final |
