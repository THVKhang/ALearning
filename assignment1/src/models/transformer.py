"""Transformer encoder classifier: image -> tokens -> self-attention stack -> CLS -> logits.

Handbook Section 11.1 asks for a token embedding or projection, a positional encoding, and an
explanation of the attention inputs and outputs.

Token embedding / projection. `src/data/sequence.py` turns the image into (N, T, D_in) raw
pixel tokens - 28 rows of 28 values, or 16 patches of 49 at the default 7x7. One `nn.Linear`
maps every token to a common width `d_model`. This is the patch-embedding step of a Vision
Transformer: attention never sees pixels directly, only these projections.

Positional encoding. Self-attention is permutation-invariant - shuffling the rows of the
input would permute the output and leave the CLS summary unchanged - so position has to be
supplied explicitly. A learned parameter of shape (1, T+1, d_model), one vector per sequence
slot, is added to the embedded tokens and trained with the rest of the model. A fixed
sinusoidal table would also work; the learned version is used here because T is small and
constant, so there is nothing to extrapolate to.

Attention inputs and outputs. Each layer maps the token sequence to three sets of vectors by
linear projections: queries Q, keys K and values V. The score between token i and token j is
the scaled dot product q_i . k_j / sqrt(d_head), and a softmax over j turns row i of the
score matrix into weights that sum to 1. The layer output at position i is that weighted sum
of value vectors. So the output at row i is a mixture of *every* row, weighted by relevance
to row i: the receptive field is global after a single layer, whereas the CNN of `cnn.py`
needs several pooling stages to span the image. `nhead` such attention maps run in parallel
over d_model / nhead channels each and are concatenated, letting one head track, say, vertical
extent while another tracks brightness.

Classification token. A learned CLS vector is prepended to the sequence. It carries no pixel
content of its own, so everything it holds after the encoder was pulled in from the image
tokens by attention; the classifier reads that single position.

Outputs raw logits: `nn.CrossEntropyLoss` applies log-softmax itself.
"""

import torch
from torch import nn

from src.data.sequence import sequence_shape, to_sequence


class TransformerClassifier(nn.Module):
    """Pre-norm encoder (`norm_first=True`), which trains at this depth without a warmup
    schedule - the post-norm arrangement of the original paper usually needs one."""

    def __init__(self, input_shape=(1, 28, 28), d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, num_classes=10, dropout=0.1, sequence="rows",
                 patch_size=7):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")

        timesteps, input_size = sequence_shape(input_shape, sequence, patch_size)
        self.sequence = sequence
        self.patch_size = patch_size
        self.timesteps = timesteps

        self.project = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positions = nn.Parameter(torch.zeros(1, timesteps + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only warns; say so
        # explicitly so a training log is not prefixed by a UserWarning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        tokens = self.project(to_sequence(x, self.sequence, self.patch_size))
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.positions
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded[:, 0]))
