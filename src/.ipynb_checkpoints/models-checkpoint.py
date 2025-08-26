import torch.nn as nn
import torch.nn.functional as F
import torch


def get_model(name: str, input_dim, output_dim, targets = None):
    if name == "linear_regression":
        return nn.Linear(input_dim, output_dim)

    if name in ["predict_mean", "predict_zero"]:
        return ConstantPredictor(targets)

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

    elif name == "cnn3d_gelu_2_6":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.6, activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_big_kernal":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5)], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_huge_kernal":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(5, 10, 10), (3, 5, 5)], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(2, 3, 3), (2, 3, 3), (2, 3, 3)], conv_channels=[32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer_big":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3)], conv_channels=[32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_fewer_channels":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5)], conv_channels=[8, 16], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer_fewer_channels":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3)], conv_channels=[8, 16, 32], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")


    elif name == "cnn3d_3_layer_1_3":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.1, p_drop_fc=.3, conv_kernel_sizes=[(2, 3, 3), (2, 3, 3), (2, 3, 3)], conv_channels=[32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer_big_1_3":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.1, p_drop_fc=.3, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3)], conv_channels=[32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer_fewer_channels_1_3":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.1, p_drop_fc=.3, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3)], conv_channels=[8, 16, 32], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_3_layer_big_0":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.0, p_drop_fc=.0, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3)], conv_channels=[32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_4_layer":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(2, 3, 3), (2, 3, 3), (2, 3, 3), (2, 3, 3)], conv_channels=[16, 32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_4_layer_1_3":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.1, p_drop_fc=.3, conv_kernel_sizes=[(2, 3, 3), (2, 3, 3), (2, 3, 3), (2, 3, 3)], conv_channels=[16, 32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_4_layer_big":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.2, p_drop_fc=.4, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3), (2, 3, 3)], conv_channels=[16, 32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    elif name == "cnn3d_4_layer_big_1_3":
        in_channels = input_dim[0]  # = 41
        return CNN3D(in_channels, output_dim, p_drop_conv=.1, p_drop_fc=.3, conv_kernel_sizes=[(3, 5, 5), (3, 5, 5), (2, 3, 3), (2, 3, 3)], conv_channels=[16, 32, 64, 128], pool_kernel_sizes=[(1, 2, 2), (1, 2, 2), (1, 2, 2), None], activation = F.gelu, approximate = "tanh")

    raise ValueError(f"undefined model name: {name}")


def get_model_input_dims(name: str):
    if name in ["linear_regression", "predict_mean", "predict_true_zero", "predict_zero"]:
        return 2
    if name in ["cnn3d", "cnn3d_dropout_0_5", "cnn3d_dropout_5_5", "cnn3d_dropout_5_0", "cnn3d_gelu_0_5", "cnn3d_gelu_2_6", "cnn3d_big_kernal", "cnn3d_huge_kernal", "cnn3d_3_layer", "cnn3d_3_layer_big", "cnn3d_fewer_channels", "cnn3d_3_layer_fewer_channels", "cnn3d_3_layer_1_3", "cnn3d_3_layer_big_1_3", "cnn3d_3_layer_fewer_channels_1_3", "cnn3d_3_layer_big_0", "cnn3d_4_layer", "cnn3d_4_layer_1_3", "cnn3d_4_layer_big", "cnn3d_4_layer_big_1_3"]:
        return 5
    else:
        raise ValueError(f"undefined model name: {name}")


class CNN3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        conv_kernel_sizes=[(2, 3, 3), (2, 3, 3)],
        conv_channels=[32, 64],
        pool_kernel_sizes=[(1, 2, 2), None],
        p_drop_conv=0,
        p_drop_fc=0,
        activation=F.relu,
        approximate=None
    ):
        super(CNN3D, self).__init__()

        assert len(conv_kernel_sizes) == len(conv_channels), \
            "conv_kernel_sizes and conv_channels must have same length"
        assert len(pool_kernel_sizes) == len(conv_channels), \
            "pool_kernel_sizes and conv_channels must have same length"

        self.layers = nn.ModuleList()
        in_ch = input_channels

        for i, (k_size, out_ch, pool_k) in enumerate(zip(conv_kernel_sizes, conv_channels, pool_kernel_sizes)):
            # Convolution
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=k_size,
                             padding=tuple(k // 2 for k in k_size))  # padding to keep dims
            bn = nn.BatchNorm3d(out_ch)
            drop = nn.Dropout3d(p_drop_conv)
            pool = nn.MaxPool3d(kernel_size=pool_k) if pool_k is not None else None

            self.layers.append(nn.ModuleDict({
                "conv": conv,
                "bn": bn,
                "drop": drop,
                "pool": pool
            }))
            in_ch = out_ch  # update for next layer

        # Global pooling + FC head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.drop_fc = nn.Dropout(p_drop_fc)
        self.fc = nn.Linear(in_ch, output_dim)

        # Activation
        if activation == F.gelu and approximate is not None:
            self.activation = lambda x: activation(x, approximate=approximate)
        else:
            self.activation = activation

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)  # (batch, channels, tod, lat, lon)

        for layer in self.layers:
            x = layer["bn"](self.activation(layer["conv"](x)))
            x = layer["drop"](x)
            if layer["pool"] is not None:
                x = layer["pool"](x)

        x = self.global_pool(x)  # (batch, channels, 1, 1, 1)
        x = self.flatten(x)      # (batch, channels)
        x = self.drop_fc(x)
        return self.fc(x)

class ConstantPredictor(nn.Module):
    def __init__(self, target_values: torch.Tensor):
        super().__init__()
        # Store as a buffer so it moves with `.to(device)` but isn’t trainable
        self.register_buffer("target_values", target_values)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.target_values.expand(batch_size, -1)
