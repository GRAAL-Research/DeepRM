from abc import abstractmethod

import torch


class DataEncoder:
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_output_dimension(self) -> int:
        ...
