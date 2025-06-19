from pathlib import Path

import math
import numpy as np
import torch
import torchvision
from torch import Tensor

from src.data.dataset.torchvision_dataset.torchvision_dataset import TorchvisionDataset
from src.data.utils import DATA_BASE_PATH

MNIST_BASE_PATH = DATA_BASE_PATH / "MNIST"


class MnistBinary(TorchvisionDataset):
    @staticmethod
    def download(data_base_path: Path, is_train_data: bool) -> tuple[Tensor, list]:
        dataset = torchvision.datasets.MNIST(root=str(data_base_path), train=is_train_data, download=True)
        return dataset.data, dataset.targets

    @staticmethod
    def get_data_base_path() -> Path:
        return DATA_BASE_PATH / "MNIST"

    @staticmethod
    def create_meta_datasets(config: dict, train: torch.Tensor, test: torch.Tensor) -> np.ndarray:
        max_digits = 10
        maximum_of_datasets = max_digits * (max_digits - 1)
        assertion_msg = f"Given {max_digits} digits, we can't create more than {maximum_of_datasets} MNIST binary datasets."
        assert config["n_dataset"] <= maximum_of_datasets, assertion_msg
        n_digits = (1 + math.sqrt(1 + 4 * int(config["n_dataset"]))) / 2
        assert n_digits.is_integer(), f"A round number of classes should be deduced from given number of datasets."
        n_digits = int(n_digits)
        assert config["n_data_per_train_dataset"] % 2 == 0 and config["n_data_per_test_dataset"] % 2 == 0, \
            "The number of instances per dataset must be even."
        n_instances_per_dataset = max(config["n_data_per_train_dataset"], config["n_data_per_test_dataset"])

        binary_datasets = []

        binary_dataset_idx = 0
        target_starting_idx = -config["target_size"]

        for first_class in range(n_digits):
            for second_class in range(n_digits):
                if len(binary_datasets) == config["n_dataset"]:
                    break
                if first_class == second_class:
                    continue

                idx = np.arange(len(train))
                np.random.shuffle(idx)
                train = train[idx]

                idx = np.arange(len(test))
                np.random.shuffle(idx)
                test = test[idx]

                if first_class not in [0, 1] and second_class not in [0, 1]:
                    first_class_filter = train[:, target_starting_idx] == first_class
                    first_class_x = train[first_class_filter, :target_starting_idx]

                    second_class_filter = train[:, target_starting_idx] == second_class
                    second_class_x = train[second_class_filter, :target_starting_idx]
                else:
                    first_class_filter = test[:, target_starting_idx] == first_class
                    first_class_x = test[first_class_filter, :target_starting_idx]

                    second_class_filter = test[:, target_starting_idx] == second_class
                    second_class_x = test[second_class_filter, :target_starting_idx]

                x = torch.vstack((first_class_x[:int(n_instances_per_dataset / 2)],
                                  second_class_x[:int(n_instances_per_dataset / 2)]))
                y = torch.ones((len(x), 1))
                y[:min(len(first_class_x), int(n_instances_per_dataset / 2))] -= 2

                binary_dataset = torch.hstack((x, y))
                if config["shuffle_each_dataset_samples"]:
                    random_indices = torch.randperm(len(x))
                    binary_dataset = binary_dataset[random_indices]

                binary_datasets.append(binary_dataset)
                binary_dataset_idx += 1

            if binary_dataset_idx == config["n_dataset"]:
                break
        smallest_n = np.inf
        for tensor in binary_datasets:
            if tensor.shape[0] < smallest_n:
                smallest_n = tensor.shape[0]

        return np.stack([tensor[:smallest_n].detach().cpu().numpy() for tensor in binary_datasets])
