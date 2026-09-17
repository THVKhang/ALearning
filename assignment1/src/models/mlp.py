"""MLP: flattened pixels -> hidden ReLU layer(s) with optional dropout -> class logits."""

from math import prod

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_sizes=(256,), num_classes=10,
                 dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = prod(input_shape)
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_features, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
