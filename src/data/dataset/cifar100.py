from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision import transforms as transforms
from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
import itertools as it
from tqdm import tqdm

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

DATA_BASE_PATH = Path("dataset")
CIFAR100_BASE_PATH = DATA_BASE_PATH / "CIFAR100"
CIFAR100_DEFAULT_IMG_SIZE = (32, 32)
CIFAR100_CACHE_BASE_PATH = CIFAR100_BASE_PATH / "cache"
NUMPY_FILE_EXTENSION = ".npy"
N_CHANNELS = 3


def load_cifar100(config: dict) -> list[Any]:
    expected_datasets_cache_path = create_datasets_cache_path(config)

    if expected_datasets_cache_path.exists():
        return np.load(expected_datasets_cache_path, allow_pickle=True)

    return create_and_store_cifar100_datasets(config)


def create_datasets_cache_path(config: dict) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        if key not in ["task", "target_size", "criterion", "is_dataset_balanced"]:
            file_name.append(f"{key}={config[key]}")

    return CIFAR100_CACHE_BASE_PATH / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def create_and_store_cifar100_datasets(config: dict) -> list[Any]:
    train, test = obtain_cifar100_dataset(config)
    datasets = create_cifar100_binary_datasets(config, train, test)
    store_cifar100_datasets(config, datasets)

    return datasets


def store_cifar100_datasets(config: dict, datasets: np.ndarray) -> None:
    CIFAR100_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config), datasets)


def create_cifar100_binary_datasets(config: dict, train, test) -> list[Tensor | Any]:
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

    return binary_datasets


def obtain_cifar100_dataset(config: dict) -> tuple[Tensor, Tensor]:
    train_set = create_train_set(config)
    test_set = create_test_set(config)

    return train_set, test_set


def create_train_set(config: dict) -> torch.Tensor:
    train_set = torchvision.datasets.CIFAR100(root=str(CIFAR100_BASE_PATH), train=True, download=True)
    train_set = apply_transforms_to_dataset(config, train_set)
    n_instances_in_cifar100_train_set = train_set.data.shape[0]

    return torch.hstack((torch.tensor(train_set.data.reshape((n_instances_in_cifar100_train_set, config["n_features"]))),
                         torch.tensor(train_set.targets).reshape(n_instances_in_cifar100_train_set, -1)))


def create_test_set(config: dict) -> torch.Tensor:
    test_set = torchvision.datasets.CIFAR100(root=str(CIFAR100_BASE_PATH), train=False, download=True)
    test_set = apply_transforms_to_dataset(config, test_set)
    n_instances_in_cifar100_test_set = test_set.data.shape[0]

    return torch.hstack((torch.tensor(test_set.data.reshape((n_instances_in_cifar100_test_set, config["n_features"]))),
                         torch.tensor(test_set.targets).reshape(n_instances_in_cifar100_test_set, -1)))


def apply_transforms_to_dataset(config: dict, dataset: CIFAR100) -> CIFAR100:
    square_root_of_n_features = np.sqrt(config["n_features"] // N_CHANNELS)
    is_a_perfect_square = 0 <= config["n_features"] // N_CHANNELS == int(square_root_of_n_features) ** 2
    assert is_a_perfect_square, "The number of features (per channel) must be a perfect square."

    new_img_size = (int(square_root_of_n_features), int(square_root_of_n_features))

    if new_img_size == CIFAR100_DEFAULT_IMG_SIZE:
        return dataset

    transform = transforms.Compose([to_pil_image, transforms.Resize(new_img_size), ToTensor()])

    transformed_data = []
    for img in tqdm(dataset.data, desc="Applying CIFAR100's transforms"):
        img = transform(img)
        transformed_data.append(img)

    dataset.data = torch.stack(transformed_data).squeeze(1)

    return dataset