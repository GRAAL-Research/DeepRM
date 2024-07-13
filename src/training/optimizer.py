import torch
from torch import nn


def create_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    if config["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])

    if config["optimizer"].lower() == "rms_prop":
        return torch.optim.RMSprop(model.parameters(), lr=config["lr"])

    raise NotImplementedError(f"The optimizer '{config['optimizer']}' is not supported.")
