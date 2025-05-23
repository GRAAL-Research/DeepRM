from abc import ABC, abstractmethod

import torch
from torch import nn as nn


class Predictor(nn.Module, ABC):
    @abstractmethod
    def __init__(self, config: dict):
        super().__init__()
        self.criterion = config["criterion"]

    @abstractmethod
    def set_params(self, params: torch.Tensor) -> None:
        ...

    @abstractmethod
    def forward(self, instances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        ...

    def _process_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.criterion == "bce_loss":
            return torch.sigmoid(output), torch.sign(output)
        elif self.criterion == "ce_loss":
            return torch.nn.Softmax(dim=2)(output), torch.nn.functional.one_hot(torch.argmax(output, dim=-1),
                                                                                num_classes=output.shape[-1])
        elif self.criterion == "mse_loss":
            return torch.relu(output), output

        raise ValueError(f"Unsupported task: {self.task}")
