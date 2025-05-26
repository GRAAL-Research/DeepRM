from abc import abstractmethod, ABCMeta
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TorchvisionDataset(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_data_base_path() -> Path:
        ...

    @staticmethod
    @abstractmethod
    def download(data_base_path: Path, is_train_data: bool) -> tuple[np.ndarray, list]:
        ...

    @staticmethod
    @abstractmethod
    def create_meta_datasets(config: dict, train: torch.Tensor, test: torch.Tensor) -> np.ndarray:
        ...
