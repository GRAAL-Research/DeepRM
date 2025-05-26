from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor

from src.data.dataset.torchvision_dataset.torchvision_dataset import TorchvisionDataset
from src.data.utils import DATA_BASE_PATH


class MnistLabelShuffle(TorchvisionDataset):
    @staticmethod
    def create_meta_datasets(config: dict, train: torch.Tensor, test: torch.Tensor) -> np.ndarray:
        binary_datasets = np.zeros((config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] +
                                    config["target_size"]))

        train_targets = train[:, -1].long()
        train = torch.hstack((train[:, :-1], F.one_hot(train_targets) * 2 - 1))
        test_targets = test[:, -1].long()
        test = torch.hstack((test[:, :-1], F.one_hot(test_targets) * 2 - 1))

        n_partition = len(train) // config["n_instances_per_dataset"]
        num_swap = 0
        stop = False
        while not stop:
            idx = np.arange(len(train))
            np.random.shuffle(idx)
            train_dataset_shuffled = train[idx].clone()
            label_idx = np.arange(784, 794)
            np.random.shuffle(label_idx)
            train_dataset_shuffled[:, -config["target_size"]:] = train_dataset_shuffled[:, label_idx]
            for num_partition in range(n_partition):
                if n_partition * num_swap + num_partition == int(
                        config["n_dataset"] * (config["splits"][0] + config["splits"][1])):
                    stop = True
                    break
                train_dataset_reduced = train_dataset_shuffled[num_partition * config["n_instances_per_dataset"]:
                                                               (num_partition + 1) * config["n_instances_per_dataset"]]
                binary_datasets[n_partition * num_swap + num_partition] = train_dataset_reduced
            num_swap += 1
        train_idx = np.arange(int(config["n_dataset"] * (config["splits"][0] + config["splits"][1])))
        np.random.shuffle(train_idx)
        binary_datasets[np.arange(len(train_idx))] = binary_datasets[train_idx]

        for binary_dataset_idx in range(int(config["n_dataset"] * (config["splits"][0] + config["splits"][1])),
                                        config["n_dataset"]):
            idx = np.arange(len(test))
            np.random.shuffle(idx)
            test_dataset_reduced = test[idx[:config["n_instances_per_dataset"]]]

            label_idx = np.arange(784, 794)
            np.random.shuffle(label_idx)
            test_dataset_reduced[:, -config["target_size"]:] = test_dataset_reduced[:, label_idx]
            binary_datasets[binary_dataset_idx] = test_dataset_reduced

        return binary_datasets

    @staticmethod
    def download(data_base_path: Path, is_train_data: bool) -> tuple[Tensor, list]:
        dataset = torchvision.datasets.MNIST(root=str(data_base_path), train=is_train_data, download=True)
        return dataset.data, dataset.targets

    @staticmethod
    def get_data_base_path() -> Path:
        return DATA_BASE_PATH / "MNIST"
