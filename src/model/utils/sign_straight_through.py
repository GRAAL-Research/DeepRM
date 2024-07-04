import torch
from torch import nn as nn


class SignStraightThrough(nn.Module):
    def __init__(self):
        """
        Implementation of the Straight through estimator.
        """
        super().__init__()

    @staticmethod
    def forward(inputs):
        """
        Forward pass for the Straight through estimator.
        Args:
            inputs (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        out = torch.sign(inputs + 1e-20) + inputs - inputs.detach()
        out[torch.abs(inputs) > 1] = out[torch.abs(inputs) > 1].detach()
        return out
