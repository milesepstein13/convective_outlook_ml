import torch.nn as nn
import torch.nn.functional as F


def get_model(name: str, input_dim, output_dim):
    if name == "linear_regression":
        return nn.Linear(input_dim, output_dim)
    elif name == "cnn3d":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim)
    raise ValueError(f"undefined model name: {name}")


def get_model_input_dims(name: str):
    if name == "linear_regression":
        return 2
    if name == "cnn3d":
        return 5
    else:
        raise ValueError(f"undefined model name: {name}")


class CNN3D(nn.Module):
    def __init__(self, input_channels: int, output_dim: int):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)  # (batch, channels, tod, lat, lon)
        x = self.bn1(F.relu(self.conv1(x)))  # (batch, 32, D-1, H, W)
        x = self.bn2(F.relu(self.conv2(x)))  # (batch, 64, D-2, H, W)
        x = self.pool(x)                     # (batch, 64, 1, 1, 1)
        x = self.flatten(x)                  # (batch, 64)
        return self.fc(x)                    # (batch, output_dim)