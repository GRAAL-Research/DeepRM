import torch
from torch import nn

from src.training.factory.message_penalty import create_message_penalty_function


def compute_loss(config: dict, criterion: nn.Module, output: torch.Tensor, targets: torch.Tensor,
                 meta_predictor) -> torch.Tensor:
    if config["is_dataset_balanced"] or config["task"] == "regression":
        loss = torch.mean(torch.mean(criterion(output, targets), dim=1) ** config["loss_exponent"])
    else:
        loss = 0
        # Loops over all of the tasks to compute the loss.
        for batch in range(len(output)):
            loss += (torch.mean(criterion(output[batch, targets[batch] == 0],
                                          targets[batch, targets[batch] == 0])) / 2 +
                     torch.mean(criterion(output[batch, targets[batch] == 1],
                                          targets[batch, targets[batch] == 1])) / 2) ** config["loss_exponent"]
        loss /= len(output)
    if config["msg_type"] == "cnt" and config["msg_size"] > 0 and config["msg_penalty_coef"] > 0:
        # There is a possibility to penalize the incurred loss with respect to the message.
        message_penalty_function = create_message_penalty_function(config)
        if meta_predictor is not None:
            loss += message_penalty_function(meta_predictor.get_message(), config["msg_penalty_coef"])

    return loss
