from abc import abstractmethod, ABCMeta

import torch
from torch import nn


class DataEncoder(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_output_dimension(self) -> int:
        ...
