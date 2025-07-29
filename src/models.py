import torch.nn as nn

def get_model(name: str, input_dim, output_dim):
    if name == "linear_regression":
        return nn.Linear(input_dim, output_dim)
    
