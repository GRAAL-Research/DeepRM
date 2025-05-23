import itertools as it
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor

from src.data.dataset.torchvision_dataset.torchvision_dataset import TorchvisionDataset
from src.data.utils import DATA_BASE_PATH


class CIFAR100(TorchvisionDataset):
    @staticmethod
    def create_meta_datasets(config: dict, train: torch.Tensor, test: torch.Tensor) -> np.ndarray:
        binary_dataset_idx = 0
        target_starting_idx = -config["target_size"]

        binary_datasets = []
        array1 = np.array(range(100))
        array2 = np.array(range(100))
        dataset_bank = np.array(list(it.product(array1, array2)))
        idx_to_rmv = []
        for dataset_idx in range(len(dataset_bank)):
            if dataset_bank[dataset_idx][0] == dataset_bank[dataset_idx][1]:
                idx_to_rmv.append(dataset_idx)
        dataset_bank = np.delete(dataset_bank, idx_to_rmv, 0)
        np.random.seed(0)
        used_idx = np.array(range(len(dataset_bank)))
        np.random.shuffle(used_idx)
        used_idx = used_idx[:150]

        for dataset_classes in used_idx:
            idx = np.arange(len(train))
            np.random.shuffle(idx)
            train = train[idx]

            idx = np.arange(len(test))
            np.random.shuffle(idx)
            test = test[idx]

            if binary_dataset_idx < 100:
                first_class_filter = train[:, target_starting_idx] == dataset_bank[dataset_classes, 0]
                first_class_x = train[first_class_filter, :target_starting_idx]

                second_class_filter = train[:, target_starting_idx] == dataset_bank[dataset_classes, 1]
                second_class_x = train[second_class_filter, :target_starting_idx]
            else:
                first_class_filter = test[:, target_starting_idx] == dataset_bank[dataset_classes, 0]
                first_class_x = test[first_class_filter, :target_starting_idx]

                second_class_filter = test[:, target_starting_idx] == dataset_bank[dataset_classes, 1]
                second_class_x = test[second_class_filter, :target_starting_idx]

            x = torch.vstack((first_class_x[:int(config["n_instances_per_dataset"] / 2)],
                              second_class_x[:int(config["n_instances_per_dataset"] / 2)]))
            y = torch.ones((len(x), 1))
            y[:min(len(first_class_x), int(config["n_instances_per_dataset"] / 2))] -= 2

            binary_dataset = torch.hstack((x, y))
            if config["shuffle_each_dataset_samples"]:
                random_indices = torch.randperm(len(x))
                binary_dataset = binary_dataset[random_indices]

            binary_datasets.append(binary_dataset)
            binary_dataset_idx += 1

            if binary_dataset_idx == config["n_dataset"]:
                break

        return np.stack([tensor.detach().cpu().numpy() for tensor in binary_datasets])

    @staticmethod
    def download(data_base_path: Path, is_train_data: bool) -> tuple[Tensor, list]:
        dataset = torchvision.datasets.CIFAR100(root=str(data_base_path), train=is_train_data, download=True)
        return dataset.data, dataset.targets

    @staticmethod
    def get_data_base_path() -> Path:
        return DATA_BASE_PATH / "CIFAR100"
