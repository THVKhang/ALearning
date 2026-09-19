from math import prod
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        input_shape=(1, 28, 28),
        hidden_sizes=(256,),
        num_classes=10,
        dropout=0.0
    ):
        super().__init__()
        layers = [nn.Flatten()]
        input_size = prod(input_shape)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    