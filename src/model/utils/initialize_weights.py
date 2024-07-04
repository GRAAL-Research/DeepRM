import torch
from torch import nn as nn


def initialize_weights(init_scheme: str, module) -> None:
    """
    Args:
        module (torch.nn.Module): The module of interest (linear layers of one or many torch.nn.Module)
    """
    for layer in module:
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data
            if init_scheme == "kaiming_unif":
                nn.init.kaiming_uniform_(weights, nonlinearity="relu")
            elif init_scheme == "kaiming_norm":
                nn.init.kaiming_normal_(weights, nonlinearity="relu")
            elif init_scheme == "xavier_unif":
                nn.init.xavier_uniform_(weights)
            elif init_scheme == "xavier_norm":
                nn.init.xavier_normal_(weights)
            else:
                raise NotImplementedError(f"The initialization method '{init_scheme}' is not supported.")
