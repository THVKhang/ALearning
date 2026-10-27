# CO3133 — Deep Learning and Its Applications
## Assignment 1 Final Report: Foundations of Deep Learning Pipelines and Architectures on Fashion-MNIST

- **Course:** CO3133 — Deep Learning and Its Applications (Semester 261)
- **Instructor:** Assoc. Prof. Dr. Lê Thành Sách
- **Institution:** Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
- **Group:** Nhóm CR7 (3 members)
  - Trần Hoàng Vỹ Khang — `2352502`
  - Đỗ Đăng Khoa — `2352554`
  - Dương Đăng Khoa — `2352551`
- **Repository:** [https://github.com/THVKhang/ALearning/tree/main/assignment1](https://github.com/THVKhang/ALearning/tree/main/assignment1)
- **Assignment Web Page:** [https://thvkhang.github.io/ALearning/assignment1/](https://thvkhang.github.io/ALearning/assignment1/)
- **Shared Course Landing Page:** [https://thvkhang.github.io/ALearning/](https://thvkhang.github.io/ALearning/)

---

## Part 1 — Problem and Data Description

### 1.1 Problem Formulation
- **Real-world motivation:** Automated garment classification underpins e-commerce catalog indexing, visual search, and retail inventory tagging, where product images frequently arrive as studio thumbnails without reliable category metadata. Methodologically, classifying low-resolution apparel imagery provides a controlled benchmark to investigate how **input data representation** (one-dimensional flattened pixel vectors vs. two-dimensional spatial grids vs. one-dimensional horizontal row sequences) and **architectural inductive bias** govern classification accuracy, parameter efficiency, and inference latency when all optimization settings are held constant.
- **Input:** A single-channel grayscale image $X \in \mathbb{R}^{1 \times 28 \times 28}$ ($784$ pixel intensities when flattened) depicting a centered article of clothing or footwear.
- **Output:** A categorical class index $\hat{y} \in \{0, 1, \dots, 9\}$ corresponding to one of 10 apparel categories (*T-shirt/top*, *Trouser*, *Pullover*, *Dress*, *Coat*, *Sandal*, *Shirt*, *Sneaker*, *Bag*, *Ankle boot*), obtained from unnormalized class logits $z = f_\theta(X) \in \mathbb{R}^{10}$.
- **Prediction unit:** One independent categorical prediction per $28 \times 28$ image.
- **Mathematical formulation:** Given a labeled training set $\mathcal{D}_{\text{train}} = \{(X_i, y_i)\}_{i=1}^{N_{\text{train}}}$, a parametric neural network $f_\theta: \mathbb{R}^{1 \times 28 \times 28} \to \mathbb{R}^{10}$ maps each image $X_i$ to unnormalized logits $z_i = f_\theta(X_i)$. The model parameters $\theta$ are optimized by minimizing the multi-class Cross-Entropy loss over the training split:
  $$\mathcal{L}(\theta) = -\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \log \frac{\exp(z_{i, y_i})}{\sum_{c=0}^{9} \exp(z_{i, c})}$$
  At inference time, the predicted category is assigned via the maximum-logit rule $\hat{y}_i = \arg\max_{c \in \{0,\dots,9\}} z_{i, c}$.
- **Main difficulties:**
  1. **Silhouette overlap among upper-body garments:** Four categories — *T-shirt/top* (0), *Pullover* (2), *Coat* (4), and *Shirt* (6) — share nearly identical outer torso boundaries and differ primarily in fine interior cues such as collars, button plackets, sleeve seams, and knit textures.
  2. **Low spatial resolution and absence of color:** Downsampling studio photographs to $28 \times 28$ single-channel grayscale eliminates chromatic cues and compresses fine structural details into 1–2 pixel intensity transitions.
  3. **Intra-class geometric variation:** Categories such as *Bag* (8), *Sandal* (5), and *Ankle boot* (9) vary substantially in strap geometry, handle orientation, and heel profile.

### 1.2 Dataset Description and Dataset Roles
In accordance with Section 10 of the Course Handbook, the role of each candidate dataset in this study is defined below:

| Dataset | Role in This Study | Scope and Usage |
| :--- | :--- | :--- |
| **MNIST** | **Development & Verification Only** | Used strictly during initial pipeline setup to verify tensor shapes and training loop integrity. No experimental claims are reported on MNIST. |
| **Fashion-MNIST** | **Primary Benchmark Dataset** | Used for **all main experimental comparisons**, exploratory data analysis, training dynamics, confusion matrices, and qualitative error analyses. |
| **CIFAR-10** | **Not Evaluated (Optional)** | Omitted in favor of a controlled **LSTM vs. GRU** recurrent architecture comparison on Fashion-MNIST under Section 11.2. |

- **Source, version, and license:** Fashion-MNIST was introduced by Xiao, Rasul, and Vollgraf (Zalando Research, 2017) under the **MIT License**.
- **Collection and labeling process:** Derived from product photographs on Zalando's e-commerce platform. Each original color image was converted to grayscale, cropped to its bounding box, resized into a $28 \times 28$ frame while preserving aspect ratio, and assigned its catalog product category label.
- **Sample counts and annotations:** $70{,}000$ images in total — $60{,}000$ official development images and $10{,}000$ official held-out test images — distributed across $10$ mutually exclusive classes.
- **Storage size and format:** Compressed IDX binary archives occupying approximately $30\text{ MB}$ on disk.
- **Bias and limitations:** Studio-lit, clean-background thumbnails (with $50.2\%$ exact-zero black background pixels) do not reflect real-world background clutter, illumination shifts, partial occlusions, or articulated human pose deformations.

### 1.3 Exploratory Data Analysis (EDA)
Our exploratory analysis of Fashion-MNIST examines input statistics, class balance, and spatial structure prior to model training:

1. **Input dimensions and pixel statistics:** Every sample is a $1 \times 28 \times 28$ grayscale raster ($784$ features when flattened) with raw 8-bit integer intensities in $[0, 255]$. When linearly rescaled to $[0.0, 1.0]$, the global pixel mean and standard deviation across the $60{,}000$ official development images are $\mu = 0.2860$ and $\sigma = 0.3530$ (recomputing these statistics solely over our $54{,}000$-image stratified training subset yields the exact same $\mu = 0.2860$ and $\sigma = 0.3530$ to four decimal places). Exactly $50.2\%$ of all pixels are $0.0$ (pure black background), yielding a strongly bimodal intensity distribution.
2. **Class distribution and imbalance analysis:** Every class $c \in \{0, \dots, 9\}$ contains **exactly $6{,}000$ official development images** ($5{,}400$ training and $600$ validation images after our stratified $90/10$ split) and **exactly $1{,}000$ official test images** ($10.0\%$ per class across all splits). This yields an **imbalance ratio of $1.0000$** ($\max_c N_c / \min_c N_c = 1.0$) and a **normalized Shannon entropy of $1.0000$**. Because the dataset is perfectly balanced, class reweighting or resampling is unnecessary, a uniform random classifier achieves $10.0\%$ accuracy, and overall Accuracy and Macro-F1 remain closely aligned.
3. **Representative samples and per-class mean templates:**
   - **Figure 1** displays a $3 \times 10$ grid of **representative training samples** (three randomly sampled training images per class, one class per column), illustrating intra-class variability in sleeve cut, surface pattern, footwear cutout, and bag handle geometry.
   - **Figure 2** displays the **per-class mean images** across the 10 categories. Visually, *Trouser*, *Bag*, *Sandal*, *Sneaker*, and *Ankle boot* each average to a sharp, distinctive silhouette, whereas *T-shirt/top*, *Pullover*, *Coat*, and *Shirt* average to nearly the same diffuse torso template — anticipating where classification errors will concentrate across all model families.

![Figure 1: Three representative training images per class (one class per column) across the 10 Fashion-MNIST categories.](assets/eda_samples_per_class.png)

![Figure 2: Per-class mean images across the 10 Fashion-MNIST categories. T-shirt/top, Pullover, Coat, and Shirt average to nearly the same torso silhouette, predicting the dominant confusion cluster.](assets/eda_class_mean_images.png)

### 1.4 Data Splitting
All six models in our benchmark share the **exact same data partitions**:
- **Split policy:** The **official test set** ($10{,}000$ images) is kept strictly untouched, while the official $60{,}000$-image development set is partitioned into a **$90\% / 10\%$ train/validation split**.
- **Random seed and stratification:** Using a deterministic pseudorandom generator initialized at seed $42$ with class stratification enabled, the $6{,}000$ development indices of each class are shuffled deterministically and partitioned into $5{,}400$ training and $600$ validation samples per class.
- **Exact sample counts:**
  - **Training split:** $54{,}000$ images ($5{,}400$ per class).
  - **Validation split:** $6{,}000$ images ($600$ per class).
  - **Test split:** $10{,}000$ images ($1{,}000$ per class).
- **Split unit and leakage prevention:** The split unit is an individual image (Fashion-MNIST contains no multi-view subject or session groupings). The training procedure never accesses the test split; test images are evaluated only once after model selection on validation Macro-F1 is finalized. Furthermore, training and validation subsets are instantiated with independent transformation pipelines so that validation samples are never altered by training-time stochastic operations. The standardization constants ($\mu = 0.2860, \sigma = 0.3530$) are identical to four decimal places on both the $54{,}000$-image training split and the $60{,}000$-image development set, with the $10{,}000$-image test set completely excluded.

### 1.5 Preprocessing and Input Representations
- **Data integrity:** All $70{,}000$ samples contain complete $28 \times 28$ pixel grids and valid integer labels in $\{0,\dots,9\}$ with no missing or corrupted entries.
- **Normalization and augmentation policy:** Raw integer pixel intensities $X \in [0, 255]$ are linearly rescaled to $[0.0, 1.0]$ and standardized channel-wise using $\mu = 0.2860$ and $\sigma = 0.3530$:
  $$\tilde{X} = \frac{X / 255.0 - 0.2860}{0.3530}$$
  Although our data pipeline supports optional training-only random horizontal flipping and random $28 \times 28$ cropping with $2$-pixel padding, data augmentation is disabled across all six main benchmark runs so that the comparison isolates architectural inductive bias without confounding from random spatial cropping.
- **Input tensor shapes and empirical batch verification:**
  - Mini-batches of size $B = 128$ are emitted as 32-bit floating-point tensors of shape $(128, 1, 28, 28)$ paired with 64-bit integer label vectors of shape $(128,)$.
  - Measured empirically on a preprocessed training batch, pixel activations exhibit a mean of $0.0073$ and a standard deviation of $0.9989$ (spanning $[-0.8102, +2.0227]$), confirming proper zero-mean unit-variance standardization.
  - Each architecture family reshapes the standardized batch $(B, 1, 28, 28)$ into its corresponding structural view:
    1. **Linear & MLP:** Flattened into a one-dimensional feature vector $(B, 784)$.
    2. **CNN:** Retained as a two-dimensional spatial grid $(B, 1, 28, 28)$.
    3. **LSTM, GRU & Transformer:** Reshaped into an ordered top-to-bottom sequence $(B, 28, 28)$ consisting of $T = 28$ horizontal row tokens, each of feature dimension $D_{\text{in}} = 28$ (vertical column serialization and $7 \times 7$ spatial patch tokenization are also supported for ablation studies).

---

## Part 2 — Methodology

### 2.1 Pipeline Overview
Our end-to-end experimental workflow follows the nine-stage structure illustrated in **Figure 3**, progressing from raw image ingestion through representation construction, forward and loss computation, iterative parameter optimization, validation-guided checkpoint selection, and held-out test evaluation.

![Figure 3: End-to-end experimental pipeline from raw Fashion-MNIST images through preprocessing, stratified mini-batch loading, model forward pass, cross-entropy loss computation, Adam optimization, maximum-logit prediction, validation checkpoint restoration, and multi-metric test evaluation.](assets/pipeline_overview.png)

Specifically, the nine stages operate as follows:
1. **Raw Data:** Ingestion of the $70{,}000$ grayscale $1 \times 28 \times 28$ Fashion-MNIST images ($[0, 255]$ intensity range across $10$ balanced apparel classes).
2. **Preprocessing:** Rescaling of pixel intensities to $[0, 1]$, standardization via $\mu = 0.2860$ and $\sigma = 0.3530$, and conversion into one of three structural representations: flattened vector $(784)$, 2D spatial grid $(1, 28, 28)$, or 28-step horizontal row sequence $(28, 28)$.
3. **Data Loader:** Deterministic stratified partitioning ($\text{seed} = 42$) into $54{,}000$ training, $6{,}000$ validation, and $10{,}000$ held-out test samples, served in mini-batches of size $B = 128$.
4. **Model ($f_\theta$):** Forward propagation through one of the six self-implemented neural architectures (Linear, MLP, CNN, LSTM, GRU, or Transformer) to produce unnormalized class logits $z \in \mathbb{R}^{B \times 10}$.
5. **Loss ($\mathcal{L}(\theta)$):** Evaluation of the multi-class Cross-Entropy objective directly on the unnormalized logits $z$ and ground-truth labels $y$.
6. **Optimization:** Backpropagation of parameter gradients $\nabla_\theta \mathcal{L}$ and weight updates using the Adam optimizer ($\eta = 10^{-3}$, zero weight decay) over a 15-epoch schedule with validation Macro-F1 tracking after every epoch.
7. **Prediction:** Deterministic forward evaluation (with dropout disabled and batch normalization statistics frozen) assigning each image to $\hat{y} = \arg\max_{c \in \{0,\dots,9\}} z_c$.
8. **Post-Processing:** Restoration of the optimal parameter state $\theta^*$ corresponding to the epoch that achieved the highest validation Macro-F1 score, followed by index-to-category mapping.
9. **Evaluation:** Comprehensive assessment on the held-out $10{,}000$-image test set across overall Accuracy, Macro-F1, Per-Class F1, the Upper-Body vs. Other-6 F1 gap, row-normalized Confusion Matrices, analytical compute (MACs/FLOPs), and inference latency.

### 2.2 Baseline and Main Methods
All six models are self-implemented from foundational neural network layers without pretrained weights or external vision backbones. Every architecture emits raw unnormalized logits $z \in \mathbb{R}^{B \times 10}$ directly to the Cross-Entropy loss function without an intermediate softmax layer during training.

#### 1. Linear / Softmax Classifier — *Baseline*
- **Input representation:** Flattened pixel vector $x \in \mathbb{R}^{B \times 784}$.
- **Architecture:** A single affine transformation mapping $\mathbb{R}^{784} \to \mathbb{R}^{10}$ via $z = xW^\top + b$ with weight matrix $W \in \mathbb{R}^{10 \times 784}$ and bias $b \in \mathbb{R}^{10}$.
- **Parameter count & compute:** $\mathbf{7{,}850}$ trainable parameters; $\mathbf{7{,}840\text{ MACs}}$ ($\approx 0.008\text{ MMACs} / 15.7\text{ KFLOPs}$) per image.
- **Strengths & limits:** Convex optimization landscape and minimal forward latency ($0.0006\text{ ms/image}$ on GPU), establishing a transparent linear lower bound. However, its decision boundaries are linear hyperplanes in raw pixel space that cannot capture pairwise pixel interactions or two-dimensional spatial locality.

#### 2. Multilayer Perceptron (MLP)
- **Input representation:** Flattened pixel vector $x \in \mathbb{R}^{B \times 784}$.
- **Architecture:** A two-layer feed-forward network with a $256$-unit hidden layer and dropout probability $p = 0.2$:
  $$\text{Flatten}(784) \longrightarrow \text{Linear}(784 \to 256) \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(p=0.2) \longrightarrow \text{Linear}(256 \to 10)$$
- **Activation & regularization rationale:** Without a nonlinear activation, stacked affine layers collapse algebraically into a single linear transformation equivalent to the baseline. The Rectified Linear Unit $\text{ReLU}(u) = \max(0, u)$ enables piecewise-linear feature composition while maintaining a unit derivative on active units to prevent gradient saturation. Dropout ($p = 0.2$) randomly zeroes $20\%$ of hidden activations during each training step, discouraging brittle co-adaptation among hidden neurons, and is disabled during evaluation.
- **Parameter count & compute:** $\mathbf{203{,}530}$ trainable parameters; $\mathbf{203{,}264\text{ MACs}}$ ($\approx 0.20\text{ MMACs} / 0.41\text{ MFLOPs}$) per image.
- **Strengths & limits:** Captures global nonlinear combinations of pixel intensities, but because the input is flattened, the network possesses no structural prior that adjacent coordinates form two-dimensional neighborhoods.

#### 3. Convolutional Neural Network (CNN)
- **Input representation:** Two-dimensional spatial grid $X \in \mathbb{R}^{B \times 1 \times 28 \times 28}$.
- **Architecture:** A self-designed two-stage convolutional feature extractor ($32$ and $64$ channels) followed by a linear classification head:
  - **Stage 1:** $3 \times 3$ Convolution ($1 \to 32$ channels, unit zero-padding) $\to$ 2D Batch Normalization $\to$ ReLU $\to$ $2 \times 2$ Max Pooling ($28 \times 28 \to 14 \times 14$).
  - **Stage 2:** $3 \times 3$ Convolution ($32 \to 64$ channels, unit zero-padding) $\to$ 2D Batch Normalization $\to$ ReLU $\to$ $2 \times 2$ Max Pooling ($14 \times 14 \to 7 \times 7$).
  - **Classification Head:** Spatial flattening ($64 \times 7 \times 7 = 3{,}136$) $\to$ Dropout ($p = 0.2$) $\to$ Linear projection ($3{,}136 \to 10$).
- **Convolution, feature maps, and pooling mechanisms:**
  - *Convolution:* Learned $3 \times 3$ kernels slide across the 2D image plane to compute localized weighted sums. Because each kernel shares its weights across all spatial positions $(i, j)$, local motifs such as collar edges, sleeve seams, and heel contours are detected translation-equivariantly using only $9 \times C_{\text{in}}$ weights per output channel. Unit padding preserves spatial resolution prior to pooling.
  - *Feature maps & normalization:* Each kernel emits a 2D activation map quantifying the presence of its learned motif across space. Stage 1 extracts $32$ low-level edge and gradient maps, whereas Stage 2 combines all $32$ maps to form $64$ higher-order texture and garment-part representations. Batch normalization stabilizes channel-wise activation distributions throughout training.
  - *Spatial pooling:* $2 \times 2$ max pooling retains the peak response within each non-overlapping $2 \times 2$ window, halving spatial dimensions at each stage ($28 \times 28 \to 14 \times 14 \to 7 \times 7$). This provides local translation invariance, reduces intermediate activation memory by $4\times$ per stage, and expands the effective receptive field of subsequent $3 \times 3$ filters.
- **Parameter count & compute:** $\mathbf{50{,}378}$ trainable parameters ($4.0\times$ fewer than the MLP); $\mathbf{3{,}869{,}824\text{ MACs}}$ ($\approx 3.87\text{ MMACs} / 7.74\text{ MFLOPs}$) per image.
- **Strengths & limits:** Encodes 2D spatial locality and translation equivariance directly into the architecture, achieving the strongest accuracy across all garment classes with a compact parameter footprint.

#### 4. Recurrent Neural Networks (LSTM & GRU) — *Core Family & Section 11.2 Extension*
- **Input representation:** Each image is serialized top-to-bottom into an ordered sequence of $T = 28$ horizontal rows $X_{\text{seq}} \in \mathbb{R}^{B \times 28 \times 28}$:
  - **Timestep ($T = 28$):** Each step $t \in \{1, \dots, 28\}$ represents one horizontal scanline progressing from the top ($t=1$, neckline/collar region) to the bottom ($t=28$, hem/sole region).
  - **Token dimension ($D_{\text{in}} = 28$):** Each input vector $x_t \in \mathbb{R}^{28}$ contains the $28$ horizontal pixel intensities of row $t$.
- **Architecture & recurrent state dynamics:**
  - A single-layer unidirectional recurrent network with hidden dimension $H = 128$ sequentially updates a hidden state $h_t \in \mathbb{R}^{128}$ from $(x_t, h_{t-1})$ across $t = 1, \dots, 28$.
  - **GRU:** Employs two gating vectors (update gate and reset gate) to interpolate between the previous state $h_{t-1}$ and a candidate recurrent activation without maintaining a separate memory cell.
  - **LSTM:** Maintains a dedicated cell state $c_t \in \mathbb{R}^{128}$ regulated by three gates (input, forget, and output gates) to propagate long-range sequential information along an additive memory pathway.
  - **Classification readout:** The terminal hidden state $h_{28} \in \mathbb{R}^{B \times 128}$, summarizing the complete top-to-bottom scan, passes through Dropout ($p = 0.2$) and a linear projection ($\mathbb{R}^{128} \to \mathbb{R}^{10}$).
- **Parameter counts & compute:**
  - **GRU:** $\mathbf{61{,}962}$ trainable parameters; $\mathbf{1{,}678{,}592\text{ MACs}}$ ($\approx 1.68\text{ MMACs} / 3.36\text{ MFLOPs}$) per image.
  - **LSTM:** $\mathbf{82{,}186}$ trainable parameters; $\mathbf{2{,}237{,}696\text{ MACs}}$ ($\approx 2.24\text{ MMACs} / 4.48\text{ MFLOPs}$) per image.
- **Strengths & limits:** Preserves horizontal within-row structure and shares recurrent transition weights across vertical scan steps, achieving high parameter efficiency. However, vertical pixel adjacency is compressed into sequential state transitions, requiring visual cues from the neckline ($t \approx 1\text{–}5$) to survive more than $20$ recurrent updates before reaching the terminal classifier at $t = 28$.

#### 5. Transformer Encoder Classifier
- **Input representation:** The same ordered sequence of $T = 28$ horizontal row tokens $X_{\text{seq}} \in \mathbb{R}^{B \times 28 \times 28}$.
- **Token projection, classification token, and positional encoding:**
  - *Row embedding projection:* A shared linear layer projects each $28$-dimensional row vector into the model dimension $d_{\text{model}} = 128$.
  - *Classification token ($\text{[CLS]}$):* A learnable embedding vector of dimension $128$ is prepended to the $28$ row embeddings, forming a sequence of length $T + 1 = 29$.
  - *Positional encoding:* Because self-attention is inherently permutation-equivariant (permuting the image rows would otherwise leave the $\text{[CLS]}$ representation unchanged), a learnable positional embedding tensor of shape $(1, 29, 128)$ is added element-wise to the token sequence before entering the encoder stack.
- **Self-attention mechanism and readout:**
  - The token sequence traverses $L = 2$ pre-norm Transformer encoder layers ($d_{\text{model}} = 128$, $H = 4$ attention heads of width $d_k = 32$ each, feed-forward dimension $d_{\text{ff}} = 256$ with GELU nonlinearity, and dropout $p = 0.2$).
  - Within each layer, multi-head self-attention projects the normalized token sequence into queries $Q$, keys $K$, and values $V$, computing scaled dot-product attention weights $\text{softmax}(QK^\top / \sqrt{32})V$. Every row token can therefore attend directly to every other row token in $O(1)$ path length from the first layer.
  - After the second encoder layer, the output representation at the prepended $\text{[CLS]}$ index undergoes Layer Normalization and a linear projection ($\mathbb{R}^{128} \to \mathbb{R}^{10}$).
- **Parameter count & compute:** $\mathbf{274{,}058}$ trainable parameters; $\mathbf{8{,}134{,}400\text{ MACs}}$ ($\approx 8.13\text{ MMACs} / 16.27\text{ MFLOPs}$) per image.
- **Strengths & limits:** Provides an immediate global receptive field across all image rows without sequential bottlenecks, but lacks built-in spatial locality priors — requiring positional relationships and local row-to-row continuity to be learned entirely from data.

### 2.3 Training Setup
To isolate the effects of input representation and architectural inductive bias, **all six models are trained under an identical experimental protocol**:
- **Optimizer & learning rate:** Adam optimizer with constant learning rate $\eta = 10^{-3}$, weight decay $\lambda = 0.0$, and no learning-rate scheduler.
- **Mini-batch size & epoch budget:** Mini-batch size $B = 128$ over a fixed budget of $15$ epochs.
- **Regularization:** Dropout with probability $p = 0.2$ in all nonlinear architectures; 2D Batch Normalization in the CNN; pre-layer Layer Normalization in the Transformer.
- **Early stopping & checkpoint selection rule:** Validation Macro-F1 is evaluated at the conclusion of every epoch with an early-stopping patience of $5$ epochs (early stopping was never triggered; all six models completed the full $15$ epochs). Whenever validation Macro-F1 improves, the corresponding model parameters $\theta^*$ are saved as the best checkpoint, which is subsequently restored for final test evaluation.
- **Reproducibility & determinism:** A global random seed of $42$ governs dataset splitting, weight initialization, mini-batch shuffling, and dropout masks, with deterministic CUDA convolution algorithms enforced across all runs.
- **Hardware & software environment:**
  - **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU ($4\text{ GB}$ VRAM, CUDA $13.0$) on Windows 11.
  - **Software:** Python $3.13.14$, PyTorch $2.14.0$, torchvision $0.29.0$, NumPy $2.5.3$, and scikit-learn $1.9.1$ using standard 32-bit floating-point precision (`fp32`). All six benchmark runs were executed back-to-back in a single controlled session.

### 2.4 Experimental Design
- **Experiment 1 — Five-Family Architectural and Representation Benchmark (Linear, MLP, CNN, LSTM/GRU, Transformer):**
  - **Hypothesis ($H_1$):** Classification performance on Fashion-MNIST is governed primarily by how well an architecture's input representation and inductive bias preserve two-dimensional spatial structure ($\text{2D Grid CNN} > \text{1D Row Sequence RNN/Transformer} > \text{1D Flattened MLP/Linear}$) rather than by raw parameter count, and inter-model performance differences will concentrate within the four visually overlapping upper-body classes (*Shirt*, *T-shirt/top*, *Pullover*, *Coat*).
  - **Manipulated factor:** Model architecture family and input representation ($(B, 784)$ flattened vector vs. $(B, 1, 28, 28)$ 2D spatial grid vs. $(B, 28, 28)$ horizontal row sequence).
  - **Controlled factors:** Dataset (Fashion-MNIST), stratified split ($54{,}000 / 6{,}000 / 10{,}000$ at seed $42$), pixel standardization ($\mu = 0.2860, \sigma = 0.3530$), Cross-Entropy loss, Adam optimizer ($\eta = 10^{-3}$, zero weight decay), batch size ($128$), epoch budget ($15$), and validation Macro-F1 checkpoint selection rule.
  - **Evaluation metrics & three-tier single-seed decision criteria:** We evaluate Test Accuracy, Test Macro-F1, Per-Class F1 (specifically the F1 gap between the four upper-body classes and the remaining six classes), trainable parameter count, analytical compute (MACs/FLOPs), training wall-clock time, and GPU vs. end-to-end inference latency. Because all benchmark runs use a single random seed ($\text{seed} = 42$), pairwise Test Macro-F1 differences ($\Delta$) are interpreted using three non-overlapping tiers:
    1. **$\Delta < 0.01$ (Practically Tied / Inconclusive under single-seed evaluation):** Differences below $1.0\text{ percentage point}$ (such as MLP $0.8902$ vs. LSTM $0.8898$ at $\Delta = 0.0004$, and LSTM $0.8898$ vs. Transformer $0.8825$ at $\Delta = 0.0073$) lie within single-seed stochastic variation and do not support superiority claims without multi-seed variance estimates.
    2. **$0.01 \le \Delta \le 0.02$ (Moderate Empirical Advantage when corroborated by parameter efficiency and per-class consistency):** Margins between $1.0$ and $2.0\text{ percentage points}$ — specifically **CNN ($0.9163$) vs. GRU ($0.9010$)** ($\Delta = +0.0153$, where the CNN leads across all $10$ classes with $18.7\%$ fewer parameters) and **GRU ($0.9010$) vs. MLP ($0.8902$)** ($\Delta = +0.0108$, where the GRU leads on $8$ of $10$ classes with $3.3\times$ fewer parameters) — represent meaningful empirical advantages in this experiment, though multi-seed runs remain desirable for formal confidence intervals.
    3. **$\Delta > 0.02$ (Large, Decisive Architectural Separation):** Margins exceeding $2.0\text{ percentage points}$ (such as CNN $0.9163$ vs. MLP $0.8902$ at $+0.0261$, CNN vs. Transformer at $+0.0338$, and all nonlinear architectures vs. Linear $0.8441$ at $+0.0384$ to $+0.0722$) reflect clear structural separations.

- **Experiment 2 — Optional Extension (Section 11.2): Controlled LSTM vs. GRU Comparison on Row Sequences:**
  - **Hypothesis ($H_2$):** Over a short $28$-step horizontal row sequence with matched hidden width ($H = 128$) and depth ($1$ layer), the two-gate **GRU** will match or exceed the three-gate **LSTM** in test Macro-F1 while requiring approximately $25\%$ fewer parameters and FLOPs.
  - **Manipulated factor:** Recurrent gating cell formulation (LSTM vs. GRU).
  - **Controlled factors:** Row sequence representation ($28$ steps $\times$ $28$ features), hidden dimension ($128$), layer depth ($1$), dropout ($p = 0.2$), linear readout head ($128 \to 10$), and all optimization hyperparameters.

---

## Part 3 — Implementation Results

### 3.1 Training Behavior
**Figure 4** presents the 15-epoch training and validation Cross-Entropy Loss and Macro-F1 trajectories across **all six architectures**.

![Figure 4: Training and validation Cross-Entropy Loss (left column) and Macro-F1 (right column) over 15 epochs for all six architectures on Fashion-MNIST (seed = 42). Vertical dotted lines mark the best validation Macro-F1 checkpoint selected for test evaluation.](assets/curves_all_models.png)

- **Convergence dynamics, underfitting vs. overfitting, and selected checkpoints:**
  - **Linear:** Exhibits **high-bias structural underfitting** relative to the nonlinear models (plateauing near $0.385$ training loss and $0.865$ training Macro-F1 because a linear hyperplane cannot partition curved feature manifolds), alongside **mild late-epoch overfitting** after **epoch 11 / 15**: validation Macro-F1 peaks at epoch 11 ($0.8536$, with validation loss $0.4186$ and training loss $0.3935$), after which training loss continues edging down to $0.3853$ at epoch 15 while validation loss drifts upward to $0.4287$ (validation Macro-F1 $0.8446$). Restoring the epoch-11 checkpoint eliminates this late degradation.
  - **MLP:** Improves monotonically across the schedule and reaches its peak validation Macro-F1 at the final epoch, **epoch 15 / 15** ($0.8991$ validation Macro-F1, $0.2887$ validation loss, $0.2145$ training loss, $0.9197$ training Macro-F1). Because validation performance is still trending upward at epoch 15, the MLP has **not fully saturated** its 15-epoch budget.
  - **CNN:** Converges most rapidly, surpassing $0.90$ validation Macro-F1 by epoch 3 ($0.9066$) and achieving its optimal checkpoint at **epoch 13 / 15** ($0.9183$ validation Macro-F1, $0.9185$ validation accuracy, $0.2323$ validation loss, $0.1562$ training loss). During epochs 14–15, training loss declines further to $0.1447$ while validation Macro-F1 dips slightly to $0.9155$, which the epoch-13 checkpoint rule cleanly avoids.
  - **LSTM & GRU:** Both recurrent architectures peak at **epoch 13 / 15** ($0.8959$ validation Macro-F1 and $0.8970$ validation accuracy for LSTM; $0.9032$ validation Macro-F1 and $0.9038$ validation accuracy for GRU) and stabilize across epochs 13–15.
  - **Transformer:** Selects its final epoch, **epoch 15 / 15** ($0.8953$ validation Macro-F1, $0.8970$ validation accuracy, $0.2949$ validation loss, $0.2695$ training loss). Both training and validation losses continue decreasing through epoch 15, indicating that the $274\text{K}$-parameter self-attention encoder **has not yet fully converged** under a short 15-epoch constant-learning-rate schedule without warmup.
- **Methodological and engineering refinements during implementation:**
  1. *Eliminating non-deterministic GPU convolution variance:* Fixing standard pseudorandom seeds reproduced the Linear, MLP, LSTM, GRU, and Transformer runs exactly, whereas runtime convolution algorithm autotuning on the GPU initially caused the CNN's peak validation Macro-F1 to fluctuate slightly between epochs 9 and 13 across repeated runs. Enforcing deterministic cuDNN convolution kernels made the CNN trajectory bit-for-bit reproducible (verified across repeated runs matching identically on training loss $0.4268 \to 0.2987 \to 0.2680$ and validation Macro-F1 $0.8763 \to 0.8950 \to 0.9066$ across the first three epochs).
  2. *Decoupling pure model inference latency from host data-loading overhead:* Timing an evaluation loop directly over a single-process data loader conflates CPU image decoding and standardization ($\sim 0.15\text{–}0.20\text{ ms/image}$) with GPU forward execution. We therefore separated latency measurement into two distinct metrics: **model-only inference latency** (measured on preloaded GPU tensors with CUDA synchronization) and **end-to-end inference latency** (including host-to-device data loading).

### 3.2 Quantitative Results
**Table 1** summarizes all primary and secondary metrics on the held-out $10{,}000$-image official test set ($\text{seed} = 42$), including exact trainable parameter counts, analytical forward-pass compute (MMACs and MFLOPs per image), training wall-clock times, and both inference latency measurements. **Table 2** reports the per-class F1 breakdown across all 10 garment categories.

**Table 1 — Overall Quantitative Comparison on the Fashion-MNIST Held-Out Test Set ($N = 10{,}000$, $\text{seed} = 42$)**

| Model | Input Representation | Params | Compute / Image (MMACs / MFLOPs) | Best Ep | Val Macro-F1 (`Val Acc`) | Test Accuracy | Test Macro-F1 | Train Time | Model-Only Inference (ms/img) | End-to-End Inference (ms/img) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear / Softmax** | Flat vector $(784)$ | **7,850** | **0.008 / 0.016** | 11 / 15 | 0.8536 (`0.8533`) | 0.8440 | 0.8441 | **125.2 s** (8.3 s/ep) | **0.0006** | 0.1992 |
| **MLP ($784\text{–}256\text{–}10$)** | Flat vector $(784)$ | 203,530 | 0.20 / 0.41 | 15 / 15 | 0.8991 (`0.8998`) | 0.8909 | 0.8902 | 137.6 s (9.2 s/ep) | 0.0015 | 0.1816 |
| **CNN ($32\text{–}64$ Conv-Pool)** | 2D grid $(1, 28, 28)$ | 50,378 | 3.87 / 7.74 | 13 / 15 | **0.9183** (`0.9185`) | **0.9165** | **0.9163** | 253.6 s (16.9 s/ep) | 0.0096 | 0.1712 |
| **LSTM ($28\text{ rows}, H=128$)** | 28 rows $\times$ 28 px | 82,186 | 2.24 / 4.48 | 13 / 15 | 0.8959 (`0.8970`) | 0.8906 | 0.8898 | 200.7 s (13.4 s/ep) | 0.0085 | 0.1538 |
| **GRU ($28\text{ rows}, H=128$)** | 28 rows $\times$ 28 px | 61,962 | 1.68 / 3.36 | 13 / 15 | 0.9032 (`0.9038`) | 0.9017 | 0.9010 | 218.9 s (14.6 s/ep) | 0.0050 | **0.1465** |
| **Transformer ($28\text{ rows}, d=128$)** | 28 row tokens + $\text{[CLS]}$ | 274,058 | 8.13 / 16.27 | 15 / 15 | 0.8953 (`0.8970`) | 0.8846 | 0.8825 | 259.2 s (17.3 s/ep) | 0.0151 | 0.1549 |

> **Single-seed interpretation ($\text{seed} = 42$):** In this benchmark run, the test Macro-F1 ranking is **CNN ($0.9163$) > GRU ($0.9010$) > MLP ($0.8902$) $\approx$ LSTM ($0.8898$) > Transformer ($0.8825$) > Linear ($0.8441$)**. Under the three-tier decision framework defined in Section 2.4: (1) margins $> 0.02$ (CNN over MLP, LSTM, Transformer, and Linear; and all nonlinear models over Linear) represent decisive structural separations; (2) margins in $[0.01, 0.02]$ (CNN over GRU by $+0.0153$ with fewer parameters, and GRU over MLP by $+0.0108$ with $3.3\times$ fewer parameters) indicate moderate empirical advantages corroborated by parameter efficiency; and (3) margins $< 0.01$ (MLP vs. LSTM at $0.0004$, and LSTM vs. Transformer at $0.0073$) fall within single-seed variance and are treated as statistically indistinguishable without multi-seed replication.

**Table 2 — Per-Class Test F1 Scores on Fashion-MNIST ($\text{seed} = 42$, ordered by Linear baseline difficulty)**

| Class | Linear | MLP | CNN | LSTM | GRU | Transformer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shirt** (`6`) | 0.5999 | 0.7138 | **0.7586** | 0.7024 | 0.7220 | 0.6636 |
| **Pullover** (`2`) | 0.7323 | 0.8091 | **0.8701** | 0.8235 | 0.8431 | 0.8129 |
| **Coat** (`4`) | 0.7442 | 0.8236 | **0.8659** | 0.8229 | 0.8419 | 0.8094 |
| **T-shirt/top** (`0`) | 0.8073 | 0.8430 | **0.8789** | 0.8326 | 0.8503 | 0.8343 |
| **Dress** (`3`) | 0.8492 | 0.8870 | **0.9117** | 0.8905 | 0.9004 | 0.8777 |
| **Sneaker** (`7`) | 0.9238 | 0.9509 | **0.9587** | 0.9439 | 0.9575 | 0.9491 |
| **Sandal** (`5`) | 0.9314 | 0.9641 | **0.9762** | 0.9619 | 0.9759 | 0.9670 |
| **Bag** (`8`) | 0.9355 | 0.9716 | **0.9814** | 0.9790 | 0.9755 | 0.9692 |
| **Ankle boot** (`9`) | 0.9499 | 0.9594 | **0.9720** | 0.9599 | 0.9630 | 0.9630 |
| **Trouser** (`1`) | 0.9672 | 0.9800 | **0.9894** | 0.9814 | 0.9804 | 0.9784 |
| **Mean F1, Upper-Body 4 (`0, 2, 4, 6`)** | 0.7209 | 0.7974 | **0.8434** | 0.7954 | 0.8143 | 0.7801 |
| **Mean F1, Other 6 (`1, 3, 5, 7, 8, 9`)** | 0.9261 | 0.9522 | **0.9649** | 0.9528 | 0.9588 | 0.9507 |
| **Gap (`Other 6 - Upper-Body 4`)** | 0.2052 | 0.1548 | **0.1215** | 0.1574 | 0.1444 | 0.1707 |
| **Macro-F1 (All 10 classes)** | **0.8441** | **0.8902** | **0.9163** | **0.8898** | **0.9010** | **0.8825** |

### 3.3 Qualitative Results and Error Analysis
**Figures 5 and 6** display representative misclassified and correctly classified test examples from the CNN model, and **Figure 7** presents the row-normalized test confusion matrices across **all six architectures**. **Table 3** quantifies the exact test error counts and relative error shares across the principal confusion clusters for all six models.

![Figure 5: CNN misclassifications on the test split (true vs. predicted category and confidence), showing errors concentrated almost entirely among visually overlapping upper-body garments.](assets/examples_incorrect_cnn.png)

![Figure 6: Correctly classified test examples from the CNN model, including both clean-silhouette items (Trouser, Bag, Sandal) and challenging long-sleeved upper-body garments (Shirt, Pullover, Coat).](assets/examples_correct_cnn.png)

![Figure 7: Row-normalized confusion matrices on the 10,000-image Fashion-MNIST test split across all six architectures (Linear, MLP, CNN, LSTM, GRU, and Transformer).](assets/confusion_matrix_all_models.png)

**Table 3 — Error Group Frequencies Across All Six Models on the Official Test Split ($N = 10{,}000$, percentages indicate shares of each model's total errors)**

| Error Group ($\text{True Category} \to \text{Predicted Category}$) | Linear | MLP | CNN | LSTM | GRU | Transformer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Test Errors (out of 10,000)** | 1,560 | 1,091 | **835** | 1,094 | 983 | 1,154 |
| **Group 1a:** *Shirt* $\to$ *T-shirt/top* | 134 (8.6%) | **106** (9.7%) | 113 (13.5%) | 161 (14.7%) | 137 (13.9%) | 146 (12.7%) |
| **Group 1b:** *T-shirt/top* $\to$ *Shirt* | 109 (7.0%) | 105 (9.6%) | **56** (6.7%) | 79 (7.2%) | 82 (8.3%) | 77 (6.7%) |
| **Group 2:** *Pullover* $\leftrightarrow$ *Coat* | 223 (14.3%) | 165 (15.1%) | **95** (11.4%) | 161 (14.7%) | 140 (14.2%) | 177 (15.3%) |
| **Group 3:** *Shirt* $\to$ *Pullover* / *Coat* | 211 (13.5%) | 143 (13.1%) | **97** (`33+64`, 11.6%) | 150 (`97+53`, 13.7%) | 137 (`75+62`, 13.9%) | 213 (`113+100`, 18.5%) |
| **All Confusions Inside Upper-Body Quartet $\{0, 2, 4, 6\}$** | 935 (59.9%) | 674 (61.8%) | **528** (63.2%) | 691 (63.2%) | 620 (63.1%) | 722 (62.6%) |
| **Group 4:** Footwear Internal $\{5, 7, 9\}$ (*Sandal*, *Sneaker*, *Ankle boot*) | 182 (11.7%) | 120 (11.0%) | **89** (10.7%) | 130 (11.9%) | 98 (10.0%) | 113 (9.8%) |
| **Group 5:** Cross-Group (Apparel $\leftrightarrow$ Footwear / Bag) | 121 (7.8%) | 53 (4.9%) | **32** (3.8%) | 38 (3.5%) | 49 (5.0%) | 58 (5.0%) |

#### Detailed Analysis of Error Groups (Description, Frequency, Examples, Hypothesized Cause, and Improvement Roadmap)
1. **Group 1 — Bidirectional *Shirt* $\leftrightarrow$ *T-shirt/top* Boundary ($169$ CNN errors up to $243$ Linear / $240$ LSTM errors):**
   - *Description & visual examples:* Short-sleeved or collarless *Shirt* instances misclassified as *T-shirt/top* (e.g., **Figure 5**, true *Shirt* predicted as *T-shirt/top* with confidence $p=0.53$) and boxy *T-shirt/top* instances misclassified as *Shirt*.
   - *Frequency:* *Shirt* $\to$ *T-shirt/top* remains within a narrow band of $106\text{–}161$ test images across all six architectures. Progressing from the Linear baseline to the CNN reduces total test errors by $46.5\%$ ($1{,}560 \to 835$), yet reduces *Shirt* $\to$ *T-shirt/top* by only $15.7\%$ ($134 \to 113$).
   - *Hypothesized cause:* This persistent error floor reflects **intrinsic visual ambiguity at $28 \times 28$ grayscale resolution**: the fine structural cues separating a casual short-sleeved shirt from a T-shirt — a collar seam or vertical button placket only 1–2 pixels wide — frequently vanish during downsampling.
   - *Concrete improvements:* Incorporate higher-resolution source imagery when available, or introduce localized contrast normalization and multi-scale ($1 \times 1$ and $3 \times 3$) convolutional branches to amplify subtle vertical placket gradients.
2. **Group 2 & Group 3 — Long-Sleeved Upper-Body Confusions (*Pullover* $\leftrightarrow$ *Coat* and *Shirt* $\to$ *Pullover* / *Coat*):**
   - *Description & visual examples:* Long-sleeved *Shirt*, *Pullover*, and *Coat* items sharing nearly identical outer torso-and-sleeve contours (e.g., **Figure 5**, true *Shirt* predicted as *Pullover* with $p=0.49$ and true *Shirt* predicted as *Coat* with $p=0.92$; whereas correctly resolved boundary cases in **Figure 6** exhibit visible neckline V-openings or hemline splits).
   - *Frequency:* This error cluster is where architectural inductive bias yields the largest separation: the CNN cuts *Pullover* $\leftrightarrow$ *Coat* confusions by $57.4\%$ relative to the Linear baseline ($223 \to 95$) and cuts *Shirt* $\to$ *Pullover* / *Coat* confusions by $54.0\%$ ($211 \to 97$, comprising $33$ *Pullover* and $64$ *Coat* predictions). Conversely, the row-tokenized **Transformer** records the highest *Shirt* $\to$ *Pullover* / *Coat* error count of all six models ($213$ errors = $113$ *Pullover* + $100$ *Coat*, representing $18.5\%$ of all its test errors).
   - *Hypothesized cause:* Distinguishing a knit pullover from a buttoned shirt or open-front coat depends on local two-dimensional surface micro-textures (knit ribbing, vertical lapel/zipper lines, and collar openings) spanning $3 \times 3$ pixel neighborhoods, which 2D convolutions extract directly but 1D horizontal row tokenization splits across separate timesteps.
   - *Concrete improvements:* (i) Replace 1D horizontal row tokens in the Transformer with **$7 \times 7$ two-dimensional spatial patch tokens** to preserve local 2D neighborhoods within each token; (ii) extend the Transformer and MLP training schedules to **30–50 epochs with linear learning-rate warmup and cosine annealing**; and (iii) investigate a **two-stage hierarchical classifier** that routes upper-body predictions to a dedicated fine-grained discriminator.
3. **Group 4 & Group 5 — Footwear Internal Confusions & Cross-Group Outliers:**
   - *Description & frequency:* Internal footwear confusions (*Sandal*, *Sneaker*, *Ankle boot*) decline from $182$ (Linear) to $89$ (CNN), occurring primarily between low-cut ankle boots and high-top sneakers ($22$ *Sneaker* $\to$ *Ankle boot* and $23$ *Ankle boot* $\to$ *Sneaker* errors in the CNN). Cross-group confusions (Apparel $\leftrightarrow$ Footwear / Bag, such as narrow vertical handbags misclassified as *Dress* in **Figure 5**) drop by $73.6\%$ from Linear ($121$) to CNN ($32$).
   - *Hypothesized cause & concrete improvements:* Minor vertical framing offsets alter the apparent shaft-to-sole aspect ratio of footwear and tall bags. Enabling training-time spatial augmentation (random horizontal flips and $2$-pixel padded random crops) should improve translation tolerance along object boundaries.

### 3.4 Comparison and Discussion
1. **Evaluation of Hypothesis $H_1$ (Partially Supported):**
   - In our controlled $\text{seed} = 42$ benchmark, Hypothesis $H_1$ is **partially supported**:
     - **Supported:** The **CNN** ranks first across every aggregate and per-class metric ($0.9163$ Macro-F1, winning all $10$ classes in Table 2) while requiring $4.0\times$ fewer parameters than the MLP and $5.4\times$ fewer than the Transformer. The **Linear** baseline ranks last ($0.8441$), and the row-recurrent **GRU** ($0.9010$) outperforms the flattened **MLP** ($0.8902$) by $+0.0108$ Macro-F1 with $3.3\times$ fewer parameters ($61{,}962$ vs. $203{,}530$). Moreover, as hypothesized, the performance spread across the five nonlinear architectures is $4.5\times$ wider on the four upper-body classes ($0.0633$ spread: $0.7801$ to $0.8434$) than on the remaining six classes ($0.0142$ spread: $0.9507$ to $0.9649$).
     - **Nuanced / Refuted aspect:** Preserving 1D horizontal row structure does **not** automatically guarantee superiority over a flattened vector representation. The **LSTM** ($0.8898$ Macro-F1) finishes virtually tied with ($0.0004$ below) the flattened **MLP** ($0.8902$), and the row-token **Transformer** ($0.8825$) trails both the MLP and the recurrent models under the fixed 15-epoch budget.
2. **Impact of Data Representation and Inductive Bias:**
   - *2D Spatial Grid (CNN, $0.9163$ Macro-F1):* Preserves both horizontal and vertical spatial axes and shares $3 \times 3$ filters across the image plane. This confirms that 2D locality and translation equivariance provide high-leverage structural priors on a medium-sized ($54{,}000$-image) dataset, reducing upper-body quartet confusions from $674$ (MLP) and $722$ (Transformer) down to $528$.
   - *1D Row Sequence (GRU $0.9010$, LSTM $0.8898$, Transformer $0.8825$):* Retains horizontal pixel adjacency within each row while encoding vertical structure through sequential transitions or positional embeddings. This yields strong parameter efficiency for the GRU ($61{,}962$ parameters) and LSTM ($82{,}186$ parameters) compared to the MLP ($203{,}530$ parameters), but does not guarantee higher accuracy when vertical features must propagate across $27$ recurrent steps (LSTM) or be inferred without spatial locality priors (Transformer).
   - *Flattened Vector (Linear $0.8441$, MLP $0.8902$):* Discards 2D topology entirely. While the linear classifier underfits, introducing a single $256$-unit ReLU hidden layer (MLP) reaches $0.8902$ Macro-F1 because Fashion-MNIST thumbnails are pre-centered and bounding-box normalized.
3. **Accuracy–Latency–Parameter–Compute Trade-offs:**
   - **Model-only GPU latency scales monotonically with analytical compute (FLOPs/MACs):** When input tensors reside in GPU memory, measured forward-pass latency follows per-image computational complexity in exact rank order:
     $$\text{Linear } (0.008\text{ MMACs}, 0.0006\text{ ms}) < \text{MLP } (0.20\text{ MMACs}, 0.0015\text{ ms}) < \text{GRU } (1.68\text{ MMACs}, 0.0050\text{ ms}) < \text{LSTM } (2.24\text{ MMACs}, 0.0085\text{ ms}) < \text{CNN } (3.87\text{ MMACs}, 0.0096\text{ ms}) < \text{Transformer } (8.13\text{ MMACs}, 0.0151\text{ ms})$$
   - **End-to-end latency is dominated by host data loading:** Under single-process CPU data loading, image decoding and standardization require $0.15\text{–}0.20\text{ ms/image}$ ($10\times$ to $330\times$ longer than the GPU forward pass), compressing end-to-end latency across all six architectures into a narrow $0.1465\text{–}0.1992\text{ ms/image}$ range.
   - **Deployment selection depends on operational constraints:** The **CNN** is optimal for classification accuracy ($0.9163$) and compact nonlinear memory footprint ($50{,}378$ parameters); the **GRU** provides the best sequential accuracy–latency balance ($0.9010$ at $0.0050\text{ ms/image}$); the **MLP** delivers $0.8902$ Macro-F1 at $6.4\times$ lower GPU latency ($0.0015\text{ ms/image}$) and $1.8\times$ shorter training time ($137.6\text{ s}$) than the CNN; and the **Linear** classifier is suited for ultra-constrained microcontroller deployment ($7{,}850$ parameters, $0.0006\text{ ms/image}$).

### 3.5 Optional Extension (Section 11.2): LSTM vs. GRU Comparison
**Table 4** compares the **LSTM** and **GRU** architectures under identical row tokenization ($28$ steps $\times$ $28$ pixels), hidden dimension ($128$), layer depth ($1$), dropout ($0.2$), and Adam optimization schedule ($\text{seed} = 42$).

**Table 4 — Controlled Recurrent Comparison: LSTM vs. GRU on Fashion-MNIST ($\text{seed} = 42$)**

| Metric | LSTM ($1\text{ layer}, H=128$) | GRU ($1\text{ layer}, H=128$) | Comparison (GRU vs. LSTM) |
| :--- | :---: | :---: | :--- |
| **Trainable Parameters** | 82,186 | **61,962** | **GRU uses $-20{,}224$ ($-24.6\%$) fewer parameters** |
| **Forward Compute (MMACs / MFLOPs)** | 2.24 / 4.48 | **1.68 / 3.36** | **GRU requires $-25.0\%$ fewer MACs/FLOPs** |
| **Total Training Time (Wall-Clock)** | **200.7 s** (13.4 s/ep) | 218.9 s (14.6 s/ep) | **LSTM trains $18.2\text{ s}$ ($8.3\%$) faster** |
| **Model-Only Inference Latency** | 0.0085 ms/img | **0.0050 ms/img** | **GRU is $41.2\%$ faster in GPU forward pass** |
| **End-to-End Inference Latency** | 0.1538 ms/img | **0.1465 ms/img** | **GRU is $4.7\%$ faster end-to-end** |
| **Training Loss (Ep 1 $\to$ Ep 13 $\to$ Ep 15)** | 0.7697 $\to$ 0.2460 $\to$ 0.2284 | **0.7579 $\to$ 0.2286 $\to$ 0.2153** | **GRU achieves lower training loss at all 15 epochs** |
| **Validation Loss (Ep 1 $\to$ Ep 13 $\to$ Ep 15)** | 0.5093 $\to$ 0.2872 $\to$ 0.2868 | **0.4863 $\to$ 0.2632 $\to$ 0.2666** | **GRU achieves lower validation loss at all 15 epochs** |
| **Training Macro-F1 (Ep 1 $\to$ Ep 13 $\to$ Ep 15)** | **0.7212** $\to$ 0.9102 $\to$ 0.9168 | 0.7210 $\to$ **0.9159 $\to$ 0.9200** | Tied at Ep 1; **GRU higher across Epochs 2–15** |
| **Validation Macro-F1 (Ep 1 $\to$ Best Ep 13 $\to$ Ep 15)** | 0.8146 $\to$ 0.8959 $\to$ 0.8951 | **0.8202 $\to$ 0.9032 $\to$ 0.9014** | **GRU achieves higher validation Macro-F1 at all 15 epochs** |
| **Test Accuracy / Test Macro-F1** | 0.8906 / 0.8898 | **0.9017 / 0.9010** | **GRU leads by $+0.0111$ Acc / $+0.0112$ Macro-F1** |

- **Balanced evaluation of Hypothesis $H_2$:**
  - **Where GRU excels:** In this $\text{seed} = 42$ comparison, the GRU achieves $+0.0112$ higher test Macro-F1 ($0.9010$ vs. $0.8898$) and outperforms the LSTM on $9$ of the $10$ garment classes (Table 2) while requiring $24.6\%$ fewer trainable parameters, $25.0\%$ fewer MACs, and $41.2\%$ lower GPU forward-pass latency ($0.0050$ vs. $0.0085\text{ ms/image}$).
  - **Where LSTM excels:** Notably, **the LSTM completed its 15 training epochs $18.2\text{ s}$ ($8.3\%$) faster in wall-clock time** ($200.7\text{ s}$ vs. $218.9\text{ s}$, or $13.4\text{ s/epoch}$ vs. $14.6\text{ s/epoch}$), attributable to fused recurrent backward-kernel scheduling on the GPU combined with single-process host data loading variance.
  - **Why GRU achieved higher Macro-F1 (faster optimization rather than LSTM overfitting):** Examining the epoch-by-epoch trajectories across all 15 epochs (Table 4) confirms that **the GRU achieved strictly lower training loss AND strictly lower validation loss than the LSTM at every single epoch from 1 to 15**, alongside strictly higher validation Macro-F1 at all 15 epochs. Consequently, the LSTM's lower test score is not caused by overfitting its larger parameter count; rather, over a short $28$-row sequence where long-horizon memory cells are unnecessary, the GRU's simpler two-gate recurrence optimizes more rapidly within a 15-epoch budget than the LSTM's three-gate plus cell-state parameterization.

### 3.6 Limitations, Generalization, and Future Work
1. **Dataset limitations & generalization to real-world retail imagery:**
   - Fashion-MNIST thumbnails are low-resolution ($28 \times 28$), single-channel grayscale, centered, and background-free ($50.2\%$ black pixels). Rankings observed on this benchmark **do not transfer unchanged to unconstrained retail photography**, where object scale, camera viewpoint, illumination, and background clutter vary substantially — in particular, the ordering $\text{CNN} > \text{Transformer}$ is characteristic of small, low-resolution datasets trained from scratch and routinely reverses on large-scale, high-resolution benchmarks with pretraining.
   - As demonstrated in Section 3.3, fine-grained categories such as *Shirt* vs. *T-shirt/top* exhibit intrinsic visual ambiguity at $28 \times 28$ grayscale resolution.
2. **Methodological and experimental limitations:**
   - **Single-seed evaluation ($\text{seed} = 42$):** Without multi-seed replication ($\text{mean} \pm \text{std}$), small margins ($< 0.01$, such as MLP vs. LSTM) are statistically indistinguishable, and moderate margins ($0.01\text{–}0.02$) lack formal confidence intervals.
   - **Fixed 15-epoch budget without architecture-specific schedule tuning:** Neither the MLP nor the Transformer had fully converged by epoch 15. Our results therefore support the specific conclusion that *"the CNN outperforms the Transformer under an equal, short 15-epoch constant-learning-rate budget without warmup,"* rather than a universal claim across all training horizons.
   - **Unmatched parameter budgets:** Trainable parameter counts span a $35\times$ range ($7{,}850$ to $274{,}058$), comparing concrete representative configurations rather than strictly parameter-matched architectures.
   - **Host-bound training wall-clock times:** Single-process data loading mixes CPU preprocessing overhead into per-epoch training times.
3. **Future work and concrete improvement roadmap:**
   - **Multi-seed replication:** Execute three independent random seeds ($42, 43, 44$) and report $\text{mean} \pm \text{std}$ across all primary and per-class metrics.
   - **Spatial patch tokenization & warmup schedule for the Transformer:** Evaluate the Transformer using $7 \times 7$ two-dimensional patch tokens over a 30–50 epoch cosine-annealing schedule with linear warmup.
   - **Parameter-matched and augmentation ablations:** Conduct controlled ablations with parameter-matched models ($\approx 60\text{K}$ parameters across families) and training-time spatial augmentation (random horizontal flipping and $2$-pixel padded random cropping).

### 3.7 Conclusion (Answer to the Research Question)
Across all five mandatory architecture families (Linear, MLP, CNN, LSTM/GRU, and Transformer) evaluated on a shared stratified Fashion-MNIST split ($54{,}000 / 6{,}000 / 10{,}000$, $\text{seed} = 42$) under an identical 15-epoch Adam training protocol, **test Macro-F1 spans from $0.8441$ (Linear) to $0.9163$ (CNN), demonstrating that classification performance is governed primarily by how well an architecture's inductive bias preserves two-dimensional local image structure rather than by total parameter count**. Specifically, our two-stage **CNN** achieves the highest Test Accuracy ($0.9165$) and Macro-F1 ($0.9163$) across all 10 garment categories using only $50{,}378$ parameters ($4.0\times$ fewer than the MLP and $5.4\times$ fewer than the Transformer), while the row-recurrent **GRU** ($0.9010$ Macro-F1, $61{,}962$ parameters) outperforms both the LSTM ($0.8898$) and the flattened MLP ($0.8902$, $203{,}530$ parameters). Crucially, $4.5\times$ more of the inter-model Macro-F1 spread occurs within the four silhouette-sharing upper-body garments (*Shirt*, *Pullover*, *Coat*, *T-shirt/top*) than across the remaining six categories, confirming that localized 2D feature extraction is the decisive differentiator on this task.

---

## AI Usage Disclosure (Handbook Section 5.2)

In full compliance with **Sections 5.1 and 5.2** of the CO3133 Course Handbook, this section documents all artificial intelligence tools used during Assignment 1 across **all three members of Group CR7**, covering all nine mandatory disclosure fields.

### 1. Group Member Roles & AI Usage Overview (All 3 Members)

| Group Member (Student ID) | Assignment 1 Contributions | AI Tools Used by Member | Verification & Ownership Role |
| :--- | :--- | :--- | :--- |
| **Trần Hoàng Vỹ Khang** (`2352502`) | Data preprocessing and splitting pipeline, training and evaluation engine, metric and visualization utilities, exploratory data analysis, integration and deterministic benchmarking of all six architectures ($\text{seed} = 42$), cross-model comparison scripts, project documentation, and final report consolidation. | **Claude Code CLI** (`Claude Sonnet 5`, `Claude Opus 5`) and **Google Antigravity (AGY) Agent** (Entries 1–4 below). | Primary verifier for all pipeline components, experimental run outputs, figures, and quantitative claims in the report. |
| **Đỗ Đăng Khoa** (`2352554`) | Implementation of the Linear and MLP baseline architectures for Milestone 1; prototype convolutional architecture exploration; verification of Part 1 (EDA and preprocessing statistics) and Part 2 (Linear, MLP, and CNN formulations, parameter counts, and analytical MACs/FLOPs). | **No generative AI tool used for authoring Milestone 1 baseline models**; participated in code review (Entries 1 & 2) and academic report refinement (Entry 4). | Co-verified tensor dimensions, parameter counts ($7{,}850$ Linear; $203{,}530$ MLP; $50{,}378$ CNN), analytical MACs/FLOPs, and EDA statistics in Parts 1 & 2. |
| **Dương Đăng Khoa** (`2352551`) | Experimental verification of sequence architectures (LSTM, GRU, and Transformer), auditing of per-class F1 scores and confusion matrices, and co-authoring and verifying Part 3 (training dynamics, error-group breakdown, controlled LSTM vs. GRU comparison, and limitations/conclusion). | **No standalone AI code generation outside shared project review sessions**; participated in reviewing Entry 3 outputs and Entry 4 report refinement. | Co-verified sequence representations ($(B, 28, 28)$), 15-epoch loss and Macro-F1 trajectories for LSTM vs. GRU, and confusion matrix error counts in Part 3. |

### 2. Detailed Disclosure Entries (All 9 Mandatory Fields of Section 5.2)

#### Entry 1 — Review and Refinement of the Milestone 1 Data and Training Pipeline
- **Tool name and version/model:** Claude Code CLI (`Claude Sonnet 5` and `Claude Opus 5`).
- **Member who used it:** Trần Hoàng Vỹ Khang (`2352502`) (reviewing implementations authored by Trần Hoàng Vỹ Khang and Đỗ Đăng Khoa).
- **Time or development stage:** 16 Sep 2026 — Milestone 1 Draft consolidation.
- **Purpose & Category (Section 5.2):** Code review, debugging, and repository hygiene verification.
- **Affected assignment section(s):** Data splitting, normalization, metric computation, plotting utilities, baseline model training loop, and **Report Part 1 (Sections 1.3–1.5)** and **Part 2 (Sections 2.1, 2.2.1, 2.2.2, 2.3)**.
- **Representative prompt examples:**
  - *"review code và làm sạch code"* (review and clean up the implementation)
  - *"kiểm tra lại phần chuẩn hoá và chia tập dữ liệu"* (re-check pixel standardization and stratified dataset splitting)
  - *"phần AI usage đã bổ sung đúng theo requirement chưa"* (verify whether the AI usage disclosure satisfies all handbook requirements)
- **How AI output was edited and verified:** The tool identified potential issues only; group members manually implemented fixes to enforce deterministic seeding, add channel standardization ($\mu = 0.2860, \sigma = 0.3530$) and class-stratified splitting, adopt an Okabe-Ito colorblind-safe plotting palette, and refine repository ignore rules.
- **Member responsible for final verification:** Trần Hoàng Vỹ Khang (`2352502`) and Đỗ Đăng Khoa (`2352554`).
- **Sources used for verification:** Official PyTorch, torchvision, and scikit-learn documentation, Course Handbook Sections 10–12, and empirical statistics computed during exploratory data analysis.

#### Entry 2 — Architecture Review of Prototype Convolutional Branch
- **Tool name and version/model:** Claude Code CLI (`Claude Sonnet 5` and `Claude Opus 5`).
- **Member who used it:** Trần Hoàng Vỹ Khang (`2352502`) (reviewing prototype branch authored by Đỗ Đăng Khoa `2352554`).
- **Time or development stage:** 22 Sep 2026 — Milestone 1 Draft consolidation.
- **Purpose & Category (Section 5.2):** Architecture review and tensor shape verification on an exploratory prototype branch.
- **Affected assignment section(s):** Exploratory prototype branch only; **no** experimental results or claims in this final report originate from that branch.
- **Representative prompt example:** *"review code của branch mlp-cnn"* (review the prototype branch implementation).
- **How AI output was edited and verified:** Verified output logit dimensions $(B, 10)$ and spatial downsampling formulas across varying convolutional block depths; the group subsequently designed a clean two-stage CNN implementation for the final benchmark.
- **Member responsible for final verification:** Trần Hoàng Vỹ Khang (`2352502`) and Đỗ Đăng Khoa (`2352554`).
- **Sources used for verification:** Course Handbook Sections 11.1 & 12.2 and official PyTorch convolutional and pooling documentation.

#### Entry 3 — Architecture Control, Determinism Debugging, and Latency Isolation for Milestone 2 Models
- **Tool name and version/model:** Claude Code CLI (`Claude Opus 5`).
- **Member who used it:** Trần Hoàng Vỹ Khang (`2352502`) (verified jointly with Đỗ Đăng Khoa `2352554` and Dương Đăng Khoa `2352551`).
- **Time or development stage:** 02–05 Oct 2026 — Milestone 2 model verification and deterministic six-model benchmark execution.
- **Purpose & Category (Section 5.2):** Architecture review, debugging of GPU convolution determinism and model-only vs. end-to-end inference timing, and cross-checking of summary tables against experimental run logs.
- **Affected assignment section(s):** CNN, LSTM, GRU, and Transformer architectures, sequence serialization, latency measurement, and **Report Part 2 (Sections 2.2.3–2.2.5, 2.3)** and **Part 3 (Sections 3.1–3.7)**.
- **Representative prompt examples:**
  - *"hãy đọc lại handbook-ene-v2 trong folder doc rồi kiểm tra kiến trúc các model M2"* (re-read the course handbook and verify the Milestone 2 model architectures)
  - *"review các model CNN, RNN, Transformer và kiểm tra tính nhất quán số liệu"* (review the CNN, RNN, and Transformer implementations and verify metric consistency)
- **How AI output was edited and verified:** Verified that all six models output raw unnormalized logits $(B, 10)$ without intermediate softmax layers; verified row, column, and patch sequence serialization element-wise; verified bit-for-bit CNN reproducibility under deterministic cuDNN settings; and cross-checked every reported metric against the saved experimental logs.
- **Member responsible for final verification:** Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`).
- **Sources used for verification:** Official PyTorch documentation for convolutional, recurrent, and Transformer modules, Course Handbook Sections 11.1–12.2, and experimental run logs.

#### Entry 4 — Generation of Combined Comparison Figures, Pipeline Diagram, and Academic Report Audit
- **Tool name and version/model:** Google Antigravity (AGY) Agent.
- **Member who used it:** Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`).
- **Time or development stage:** Oct 2026 — Milestone 2 Final report refinement, visual diagram generation, and scientific audit.
- **Purpose & Category (Section 5.2):** Generation of the end-to-end pipeline overview diagram (**Figure 3**), combined six-model training curves (**Figure 4**), and combined six-model confusion matrices (**Figure 7**); editorial and numerical audit of the academic prose to ensure all tables, error counts, analytical MACs/FLOPs, and scientific conclusions adhere to academic reporting standards.
- **Affected assignment section(s):** **Figures 3, 4, and 7**, and **Parts 1, 2, 3, and AI Usage Disclosure** of this report.
- **Representative prompt examples:**
  - *"kiểm tra lại số liệu giữa các bảng và báo cáo xem đã khớp hoàn toàn với kết quả thực nghiệm chưa"* (reconcile all reported numbers across tables and prose against the experimental outputs)
  - *"vẽ sơ đồ pipeline overview và chỉnh lại văn phong chuẩn báo cáo học thuật, hạn chế ghi tên file code hay tên biến"* (draw the pipeline overview diagram and refine the prose into formal academic style without code filenames or variable names)
- **How AI output was edited and verified:** Cross-checked every cell of Tables 1–4 against the experimental evaluation logs (confirming confusion matrix entries, epoch-by-epoch GRU vs. LSTM training and validation trajectories across all 15 epochs, and analytical MACs/FLOPs per image) and visually inspected all generated figures for layout clarity and typographical accuracy.
- **Member responsible for final verification:** Trần Hoàng Vỹ Khang (`2352502`), Đỗ Đăng Khoa (`2352554`), and Dương Đăng Khoa (`2352551`).
- **Sources used for verification:** Saved experimental evaluation logs, Course Handbook Sections 3.1–3.3, 5.2, and 10–13.
