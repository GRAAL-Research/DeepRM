from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor

from src.data.dataset.torchvision_dataset.torchvision_dataset import TorchvisionDataset
from src.data.utils import DATA_BASE_PATH


class MnistBinaryWithOnlyTwoClasses(TorchvisionDataset):
    @staticmethod
    def download(data_base_path: Path, is_train_data: bool) -> tuple[Tensor, list]:
        dataset = torchvision.datasets.MNIST(root=str(data_base_path), train=is_train_data, download=True)
        return dataset.data, dataset.targets

    @staticmethod
    def get_data_base_path() -> Path:
        return DATA_BASE_PATH / "MNIST"

    @classmethod
    def create_meta_datasets(cls, config: dict, train: torch.Tensor, test: torch.Tensor) -> np.ndarray:
        train = cls.preprocess_set_for_binary(config, train)
        test = cls.preprocess_set_for_binary(config, test)
        n_instances_per_dataset = max(config["n_data_per_train_dataset"], config["n_data_per_test_dataset"])
        binary_datasets = np.zeros((config["n_dataset"], n_instances_per_dataset, config["n_features"] +
                                    config["target_size"]))
        n_partition = len(train) // n_instances_per_dataset
        num_swap = 0
        stop = False
        while not stop:
            idx = np.arange(len(train))
            np.random.shuffle(idx)
            train_dataset_shuffled = train[idx].clone()
            for pixel in range(config["n_pixels_to_permute"]):
                first_pixel_location = np.random.randint(0, config["n_features"])
                second_pixel_location = np.random.randint(0, config["n_features"])
                first_pixel = train_dataset_shuffled[:, first_pixel_location].clone()
                second_pixel = train_dataset_shuffled[:, second_pixel_location].clone()
                train_dataset_shuffled[:, second_pixel_location] = first_pixel
                train_dataset_shuffled[:, first_pixel_location] = second_pixel
            for num_partition in range(n_partition):
                if n_partition * num_swap + num_partition == int(
                        config["n_dataset"] * (config["splits"][0] + config["splits"][1])):
                    stop = True
                    break
                train_dataset_reduced = train_dataset_shuffled[num_partition * n_instances_per_dataset:
                                                               (num_partition + 1) * n_instances_per_dataset]
                binary_datasets[n_partition * num_swap + num_partition] = train_dataset_reduced
            num_swap += 1
        train_idx = np.arange(int(config["n_dataset"] * (config["splits"][0] + config["splits"][1])))
        np.random.shuffle(train_idx)
        binary_datasets[np.arange(len(train_idx))] = binary_datasets[train_idx]
        for binary_dataset_idx in range(int(config["n_dataset"] * (config["splits"][0] + config["splits"][1])),
                                        config["n_dataset"]):
            idx = np.arange(len(test))
            np.random.shuffle(idx)
            test_dataset_reduced = test[idx[:n_instances_per_dataset]]

            for pixel in range(config["n_pixels_to_permute"]):
                first_pixel_location = np.random.randint(0, config["n_features"])
                second_pixel_location = np.random.randint(0, config["n_features"])
                first_pixel = test_dataset_reduced[:, first_pixel_location].clone()
                second_pixel = test_dataset_reduced[:, second_pixel_location].clone()
                test_dataset_reduced[:, second_pixel_location] = first_pixel
                test_dataset_reduced[:, first_pixel_location] = second_pixel
            binary_datasets[binary_dataset_idx] = test_dataset_reduced
        return binary_datasets

    @staticmethod
    def preprocess_set_for_binary(config: dict, dataset: torch.Tensor) -> torch.Tensor:
        features = dataset[:, :config["n_features"]]
        targets = dataset[:, config["n_features"]:]
        first_digit_indices = (targets == config["first_digit"]).squeeze()
        second_digit_indices = (targets == config["second_digit"]).squeeze()
        first_digit_features = features[first_digit_indices]
        second_digit_features = features[second_digit_indices]

        target_first = torch.ones((first_digit_indices.sum(), 1)) * -1
        target_second = torch.ones((second_digit_indices.sum(), 1))

        first_digit_instances = torch.hstack((first_digit_features, target_first))
        second_digit_instances = torch.hstack((second_digit_features, target_second))
        all_instances = torch.vstack((first_digit_instances, second_digit_instances))

        return all_instances[torch.randperm(all_instances.size()[0])]
