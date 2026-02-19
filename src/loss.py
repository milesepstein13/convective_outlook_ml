"""Losses for training.

Classes
---------
ShashNLL(torch.nn.Module)

"""

import torch
import pandas as pd
import numpy as np
from src.shash_torch import Shash


class ShashNLL(torch.nn.Module):
    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, outputs, target):
        """
        outputs: (B, 4*K)
        target:  (B, K)
        """
        B = outputs.shape[0]
        K = target.shape[1] if target.ndim > 1 else 1

        params = outputs.view(B, K, 4)

        mu    = params[..., 0]
        sigma = torch.exp(params[..., 1])
        gamma = params[..., 2]
        tau   = torch.exp(params[..., 3])

        params_valid = torch.stack([mu, sigma, gamma, tau], dim=-1)

        dist = Shash(params_valid)

        loss = -torch.log(dist.prob(target) + self.eps)  # (B, K)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")