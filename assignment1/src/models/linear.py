"""Linear / softmax classifier: flattened pixels -> one linear layer -> class logits.

Outputs raw logits on purpose: nn.CrossEntropyLoss applies log-softmax internally,
so applying softmax here would count it twice.
"""

from math import prod

from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(input_shape), num_classes),
        )

    def forward(self, x):
        return self.net(x)
