import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=True):
    """Seed every RNG the pipeline draws from, and optionally pin cuDNN.

    Seeding alone is not enough to reproduce a convolutional model. cuDNN picks
    a convolution algorithm at run time and some of its choices accumulate
    gradients with atomics, so two runs with the same seed on the same GPU can
    diverge. Measured here: the CNN moved from best epoch 9 (val macro-F1
    0.9208) to best epoch 13 (0.9202) across two such runs, while the Linear
    and MLP models - which have no convolution - reproduced exactly.

    deterministic=True forces cuDNN to deterministic algorithms and disables
    its autotuner, which costs some convolution throughput and buys a run that
    can actually be reproduced from the recorded seed (handbook Section 4.2).

    torch.use_deterministic_algorithms is deliberately not used: it raises
    unless CUBLAS_WORKSPACE_CONFIG is set, and cuDNN has no deterministic RNN
    backward, so it would make the LSTM and GRU runs fail outright.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    print(f"[seed] set to {seed} (cudnn deterministic={deterministic})")

