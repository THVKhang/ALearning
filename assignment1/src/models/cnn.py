"""CNN: stacked conv-pool blocks over the 28x28 image -> flattened feature maps -> logits.

Why a CNN and not the flattened MLP of `mlp.py`: flattening throws away the fact that two
neighbouring pixels belong to the same part of the garment. Three ideas recover that, and
handbook Section 11.1 asks for each to be explained.

Convolution. A conv layer slides a small learned kernel (3x3 here) over its input and takes
the weighted sum under the window. The same kernel is reused at every position, so a pattern
such as "a vertical edge" is detected wherever it appears, and one output channel costs
`3*3*C_in` weights instead of one weight per pixel position as in a linear layer. With
`padding=1` a 3x3 kernel leaves the height and width unchanged, so the only downsampling in
this model comes from the pooling layers.

Feature maps. One kernel produces one output image - a feature map - whose value at (i, j)
measures how strongly that kernel's pattern is present around pixel (i, j). A block with
`out_channels=32` therefore emits 32 feature maps stacked as channels. The next block
convolves across all of them at once, so its kernels combine simple patterns into composite
ones: edges into corners and textures, then into sleeve, heel or strap shapes.

Pooling. `MaxPool2d(2)` keeps only the largest value in each 2x2 window, halving height and
width. It discards where exactly inside the window the feature fired while keeping that it
fired, which buys a little translation tolerance, cuts the activation tensor 4x, and doubles
the area of the original image that the next 3x3 kernel can see.

Outputs raw logits, like the other models here: `nn.CrossEntropyLoss` applies log-softmax
internally, so a softmax at this point would be applied twice.
"""

from torch import nn


class CNN(nn.Module):
    """One Conv-BatchNorm-ReLU-MaxPool block per entry in `channels`.

    BatchNorm2d normalizes each feature map over the batch, which keeps the activation
    scale stable across blocks and lets the same learning rate train a deeper stack than
    plain Conv-ReLU would.
    """

    def __init__(self, input_shape=(1, 28, 28), channels=(32, 64), num_classes=10,
                 dropout=0.0):
        super().__init__()
        in_channels, height, width = input_shape

        blocks = []
        for out_channels in channels:
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Each MaxPool2d(2) floor-divides the spatial size, so n blocks turn a 28x28 input
        # into 28 // 2**n per side: 2 blocks -> 7x7, 3 -> 3x3, 4 -> 1x1.
        pooled_height = height // 2 ** len(channels)
        pooled_width = width // 2 ** len(channels)
        if min(pooled_height, pooled_width) < 1:
            raise ValueError(f"{len(channels)} pooling blocks shrink a {height}x{width} "
                             "input below one pixel")

        head = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(in_channels * pooled_height * pooled_width, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.features(x))
