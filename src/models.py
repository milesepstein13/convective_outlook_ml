import torch.nn as nn


def get_model(name: str, input_dim, output_dim):
    if name == "linear_regression":
        return nn.Linear(input_dim, output_dim)
    raise ValueError(f"undefined model name: {name}")


def get_model_input_dims(name: str):
    if name == "linear_regression":
        return 2
    else:
        raise ValueError(f"undefined model name: {name}")