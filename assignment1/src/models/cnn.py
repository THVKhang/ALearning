from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        input_shape=(1, 28, 28),
        channels=(32, 64),
        classifier_hidden=128,
        num_classes=10,
        dropout=0.0,
    ):
        super().__init__()

        # Feature extractor
        feature_layers = []
        input_channels = input_shape[0]

        for channel in channels:
            feature_layers.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                )
            )
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.MaxPool2d(kernel_size=2))

            input_channels = channel

        self.features = nn.Sequential(*feature_layers)

        # Spatial size after pooling layers
        height = input_shape[1] // (2 ** len(channels))
        width = input_shape[2] // (2 ** len(channels))

        flattened_size = channels[-1] * height * width

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x