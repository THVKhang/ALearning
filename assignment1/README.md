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
| M1 Draft (25% of A1) | 23 Sep 2026 | EDA, Dataset/DataLoader, train/val loop, Linear + MLP runnable | **code complete** |
| M2 Final (75% of A1) | 21 Oct 2026 | all five models, full comparison, report, slides, video, page, checkpoints | not started |

Still outstanding for M1 submission: report PDF, slides, YouTube video, and the
Assignment 1 web page. CNN, LSTM/GRU and Transformer belong to M2 Final; their module
files exist but are intentionally empty.

## Project layout

```
assignment1/
├── configs/default.yaml        experiment configuration (seed, data, model, training)
├── src/
│   ├── data/dataset.py         Fashion-MNIST/MNIST loading, stratified split, DataLoaders
│   ├── data/sequence.py        image -> sequence for LSTM/Transformer (M2, empty)
│   ├── models/linear.py        linear / softmax classifier
│   ├── models/mlp.py           multilayer perceptron
│   ├── models/{cnn,rnn,transformer}.py   M2 Final (empty)
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
per-class F1, parameter count, inference time, confusion matrix),
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

## Draft results (Fashion-MNIST, seed 42)

Held-out official test split, 10,000 images. Checkpoint = best validation macro-F1.

| Model | Params | Test accuracy | Test macro-F1 | Best epoch | Train time | Inference |
|---|---|---|---|---|---|---|
| Linear / softmax | 7,850 | 0.8440 | 0.8441 | 11 / 15 | 325.0 s (21.7 s/epoch) | 0.354 ms/image |
| MLP (784-256-10, dropout 0.2) | 203,530 | 0.8909 | 0.8902 | 15 / 15 | 301.6 s (20.1 s/epoch) | 0.350 ms/image |

Accuracy and macro-F1 agree to within 0.001 for both models, which is what the balanced
class distribution predicts.

### Per-class F1 on the test split

| Class | Linear | MLP | Gain |
|---|---|---|---|
| Shirt | 0.5999 | 0.7138 | +0.114 |
| Pullover | 0.7323 | 0.8091 | +0.077 |
| Coat | 0.7442 | 0.8236 | +0.079 |
| T-shirt/top | 0.8073 | 0.8430 | +0.036 |
| Dress | 0.8492 | 0.8870 | +0.038 |
| Sneaker | 0.9238 | 0.9509 | +0.027 |
| Sandal | 0.9314 | 0.9641 | +0.033 |
| Bag | 0.9355 | 0.9716 | +0.036 |
| Ankle boot | 0.9499 | 0.9594 | +0.010 |
| Trouser | 0.9672 | 0.9800 | +0.013 |

### What the draft comparison does and does not show

- The hidden layer buys **+4.6 macro-F1 points for 26x the parameters**. The gain is not
  spread evenly: it averages **+0.076 on the four upper-body garments** (Shirt, Pullover,
  Coat, T-shirt/top) against **+0.026 on the other six classes**, roughly a 3x
  difference.
- That split was predicted by the EDA before training: those four classes share almost
  the same silhouette and differ in local texture, so a linear decision boundary over raw
  pixels cannot separate them, while every class with a distinctive global shape
  (Trouser, Bag, Sneaker, Ankle boot) is already above 0.92 F1 for the linear model. This
  is the concrete form the inductive-bias argument takes on this dataset, and it is the
  hypothesis the CNN in M2 should push further.
- **Shirt is the dominant error group** for both models, confusing with T-shirt/top,
  Pullover and Coat (see `confusion_matrix_test.png` in each run directory).
- **The MLP is not converged.** Its best epoch is the last one in the budget (15/15) and
  validation macro-F1 was still rising, so 0.8902 is a lower bound for this architecture,
  not its ceiling. The linear model peaked at epoch 11 with validation loss drifting up
  while training loss kept falling - mild overfitting, absorbed by the checkpoint rule.
- **No speed claim can be made from this table.** The two timing columns are within 2-8%
  of each other because both are bound by data loading, not computation (see the caveat
  below). The MLP run is additionally contaminated: the EDA notebook was executing on the
  same machine during its epochs 5-7, which is why those epochs are visibly faster in
  `history.json` (another process released the CPU). Timing must be re-measured under a
  quiet machine and a loader-independent protocol before it goes in the report.



## Reproducibility

| Item | Value |
|---|---|
| Seed | 42, applied to `random`, `numpy`, `torch`, `torch.cuda` (`src/utils/seed.py`) |
| Split | stratified 90/10 of the official training set; official test set untouched during training |
| Split unit | one image (no grouping structure in Fashion-MNIST) |
| Checkpoint rule | best validation macro-F1; early stopping after 5 epochs without gain |
| Optimizer | Adam, lr 1e-3, weight decay 0, batch size 128, max 15 epochs |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11 |
| Software | Python 3.13.14, torch 2.14.0+cu130, torchvision 0.29.0+cu130, numpy 2.5.3, scikit-learn 1.9.1 |
| Mixed precision | not used |

Every number reported above is traceable to a run directory under `outputs/`: the model
configuration and library versions are in its `config.json`, the per-epoch curve data in
`history.json`, the test metrics in `metrics_test.json`, and the weights in `best.pt`.

**Caveat on the timing columns.** With `num_workers: 0` the pipeline is bound by
single-process CPU preprocessing, not by the model. For Linear and MLP the reported
training and inference times therefore mostly measure the data loader, and they are only
comparable between these two runs, which share the identical loader configuration. The
M2 comparison against CNN/LSTM/Transformer needs either a higher `num_workers` or a
timing method that excludes data loading.

## Deliverables

- Source code: this directory
- AI Usage Disclosure: [AI_USAGE.md](AI_USAGE.md) - **incomplete, see the TODO fields**
- Assignment 1 page: `index.html` (to be written)
- Report, slides, YouTube video: to be produced
- Checkpoints: `outputs/<run_name>/best.pt`, reproducible with the commands above
