from abc import abstractmethod

import torch
from torch import nn


class DataEncoder(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_output_dimension(self) -> int:
        raise NotImplementedError
