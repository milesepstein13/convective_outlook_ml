import torch.nn as nn
import torch.nn.functional as F


def get_model(name: str, input_dim, output_dim):
    if name == "linear_regression":
        return nn.Linear(input_dim, output_dim)
    elif name == "cnn3d":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim)
    elif name == "cnn3d_dropout_0_5":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_fc=.5)

    elif name == "cnn3d_dropout_5_5":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.5, p_drop_fc=.5)

    elif name == "cnn3d_dropout_5_0":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.5)

    elif name == "cnn3d_gelu_0_5":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_fc=.5, activation = F.gelu, approximate = "tanh")
    raise ValueError(f"undefined model name: {name}")


def get_model_input_dims(name: str):
    if name == "linear_regression":
        return 2
    if name in ["cnn3d", "cnn3d_dropout_0_5", "cnn3d_dropout_5_5", "cnn3d_dropout_5_0", "cnn3d_gelu_0_5"]:
        return 5
    else:
        raise ValueError(f"undefined model name: {name}")


class CNN3D(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, p_drop_conv=0, p_drop_fc=0, activation=F.relu, approximate = None):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.drop1 = nn.Dropout3d(p_drop_conv)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.drop2 = nn.Dropout3d(p_drop_conv)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.drop_fc = nn.Dropout(p_drop_fc)

        self.fc = nn.Linear(64, output_dim)

        if activation == F.gelu and approximate is not None:
            self.activation = lambda x: activation(x, approximate=approximate)
        else:
            self.activation = activation


    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)         # (batch, channels, tod, lat, lon)
        x = self.bn1(self.activation(self.conv1(x)))  # (batch, 32, D-1, H, W)
        x = self.bn2(self.activation(self.conv2(x)))  # (batch, 64, D-2, H, W)
        x = self.pool(x)                     # (batch, 64, 1, 1, 1)
        x = self.flatten(x)                  # (batch, 64)
        x = self.drop_fc(x)                  # fc dropout
        return self.fc(x)                    # (batch, output_dim)