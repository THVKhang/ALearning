"""LSTM / GRU classifier: the image is read as a token sequence, final hidden state -> logits.

Handbook Section 11.1 asks for the timestep, the input size and the hidden representation to
be explained.

Timestep. With the default `sequence="rows"` each of the 28 image rows is one timestep, so
the model reads the garment top to bottom in 28 steps. `columns` reads it left to right;
`patches` reads 7x7 blocks in row-major order, 16 steps. `src/data/sequence.py` holds the
conversion and is shared with the Transformer, so both sequence models see identical tokens.

Input size. The length of one token: `channels * width` = 28 values for a row of the
1-channel 28x28 image, or `channels * patch_size**2` = 49 for a 7x7 patch. This is the
`input_size` argument of `nn.LSTM` / `nn.GRU`.

Hidden representation. The cell carries a hidden state `h_t` of `hidden_size` values and
rewrites it at every step from the pair (x_t, h_{t-1}). A GRU does this with two gates -
update and reset - deciding how much of the previous state to keep and how much of the new
candidate to write. An LSTM adds a separate cell state `c_t` and a third gate, so it can
carry information along `c_t` almost unchanged across many steps while the output gate
controls what of it is exposed as `h_t`; that extra path is what makes long sequences
trainable. Either way `h_t` after the last timestep is a fixed-size summary of the whole
image, and the classifier reads only that vector - not the per-timestep outputs.

Note the contrast with `cnn.py`: a CNN shares weights across *space* and sees a local
neighbourhood that grows with depth, while a recurrent model shares weights across *time*
and must pass information about row 1 through 27 updates before it can affect the prediction.

`cell="lstm"` and `cell="gru"` select the two mandatory variants; training both is also the
optional LSTM-vs-GRU comparison of Section 11.2.

Outputs raw logits: `nn.CrossEntropyLoss` applies log-softmax itself.
"""

from torch import nn

from src.data.sequence import sequence_shape, to_sequence


class RecurrentClassifier(nn.Module):
    """`dropout` is applied between stacked layers (only when `num_layers` > 1, which is
    what `nn.LSTM` supports) and again to the summary vector before the classifier."""

    CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, input_shape=(1, 28, 28), cell="lstm", hidden_size=128, num_layers=1,
                 num_classes=10, dropout=0.0, sequence="rows", patch_size=7):
        super().__init__()
        if cell not in self.CELLS:
            raise ValueError(f"unknown cell: {cell} (expected lstm or gru)")

        timesteps, input_size = sequence_shape(input_shape, sequence, patch_size)
        self.sequence = sequence
        self.patch_size = patch_size
        self.timesteps = timesteps

        self.rnn = self.CELLS[cell](
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head = [nn.Dropout(dropout)] if dropout > 0 else []
        head.append(nn.Linear(hidden_size, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        tokens = to_sequence(x, self.sequence, self.patch_size)
        # `outputs[:, -1]` is the top-layer hidden state at the final timestep, which both
        # nn.LSTM and nn.GRU expose the same way - no need to unpack their differing states.
        outputs, _ = self.rnn(tokens)
        return self.classifier(outputs[:, -1])
