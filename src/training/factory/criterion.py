from torch import nn


def create_criterion(config: dict) -> nn.Module:
    if config["criterion"].lower() == "bce_loss":
        return nn.BCELoss(reduction="none")

    elif config["criterion"].lower() == "ce_loss":
        return nn.CrossEntropyLoss(reduction="none")

    elif config["criterion"].lower() == "mse_loss":
        return nn.MSELoss(reduction="none")

    raise NotImplementedError(f"The criterion '{config['criterion']}' is not supported.")
