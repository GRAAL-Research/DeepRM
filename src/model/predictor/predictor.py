from abc import ABC, abstractmethod

import torch
from torch import nn as nn


class Predictor(nn.Module, ABC):
    @abstractmethod
    def __init__(self, config: dict):
        super().__init__()
        self.task = config["task"]

    @abstractmethod
    def set_weights(self, weights: torch.Tensor) -> None:
        ...

    @abstractmethod
    def forward(self, instances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        ...

    def _process_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task == "classification":
            return torch.sigmoid(output), torch.sign(output)
        elif self.task == "regression":
            return torch.relu(output), output

        raise ValueError(f"Unsupported task: {self.task}")
