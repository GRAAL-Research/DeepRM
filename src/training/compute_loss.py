import torch
from torch import nn

from src.training.factory.message_penalty import create_message_penalty_function


def compute_loss(config: dict, criterion: nn.Module, output: torch.Tensor, targets: torch.Tensor,
                 meta_predictor) -> torch.Tensor:
    if config["is_dataset_balanced"] or config["task"] == "regression":
        loss = torch.mean(torch.mean(criterion(output, targets), dim=0) ** config["loss_exponent"])
    else:
        loss = 0
        for batch in range(len(output)):
            loss += (torch.mean(criterion(output[batch, targets[batch] == 0],
                                          targets[batch, targets[batch] == 0])) / 2 +
                     torch.mean(criterion(output[batch, targets[batch] == 1],
                                          targets[batch, targets[batch] == 1])) / 2) ** config[
                        "loss_exponent"]
        loss /= len(output)
    if config["msg_type"] is not None and config["msg_size"] > 0:
        message_penalty_function = create_message_penalty_function(config)
        if meta_predictor is not None:
            loss += message_penalty_function(meta_predictor.get_message(), config["msg_penalty_coef"])

    return loss
