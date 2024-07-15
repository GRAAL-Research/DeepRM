import torch
from torch.optim import Optimizer, lr_scheduler


def create_scheduler(config: dict, optimizer: Optimizer) -> lr_scheduler:
    if config["scheduler"].lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max",
                                                          factor=config["scheduler_factor"],
                                                          patience=config["scheduler_patience"],
                                                          threshold=config["scheduler_threshold"],
                                                          verbose=True)

    raise NotImplementedError(f"The scheduler '{config['scheduler']}' is not supported.")
