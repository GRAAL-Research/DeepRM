from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import math

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

DATA_BASE_PATH = Path("dataset")
CIFAR100_BASE_PATH = DATA_BASE_PATH / "CIFAR100"
CIFAR100_DEFAULT_IMG_SIZE = (32, 32)
CIFAR100_CACHE_BASE_PATH = CIFAR100_BASE_PATH / "cache"
NUMPY_FILE_EXTENSION = ".npy"
N_CHANNELS = 3


def load_cifar100(config: dict) -> np.ndarray:
    expected_datasets_cache_path = create_datasets_cache_path(config)

    if expected_datasets_cache_path.exists():
        return np.load(expected_datasets_cache_path)

    return create_and_store_cifar100_datasets(config)


def create_datasets_cache_path(config: dict) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        if key not in ["task", "target_size", "criterion", "is_dataset_balanced"]:
            file_name.append(f"{key}={config[key]}")

    return CIFAR100_CACHE_BASE_PATH / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def create_and_store_cifar100_datasets(config: dict) -> np.ndarray:
    dataset = obtain_cifar100_dataset(config)
    datasets = create_cifar100_binary_datasets(config, dataset)
    store_cifar100_datasets(config, datasets)

    return datasets


def store_cifar100_datasets(config: dict, datasets: np.ndarray) -> None:
    CIFAR100_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config), datasets)


def create_cifar100_binary_datasets(config: dict, dataset) -> np.ndarray:
    n_classes = 100
    max_of_datasets = n_classes * (n_classes - 1)
    assertion_msg = f"Given {n_classes} digits, we can't create more than {max_of_datasets} CIFAR100 binary datasets."
    assert config["n_dataset"] <= max_of_datasets, assertion_msg
    assert config["n_instances_per_dataset"] % 2 == 0, "The number of instances per dataset must be even."

    binary_datasets = np.zeros(
        (config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] + config["target_size"]))

    n_instance_per_class_per_dataset = config["n_instances_per_dataset"] // 2
    target_starting_idx = -config["target_size"]

    n_train_datasets = math.floor(config["n_dataset"] * config["splits"][0])
    n_valid_datasets = math.floor(config["n_dataset"] * config["splits"][1])
    n_test_datasets = math.floor(config["n_dataset"] * config["splits"][2])

    n_train_classes = round(n_train_datasets ** 0.5) + 1
    n_valid_classes = round(n_valid_datasets ** 0.5) + n_train_classes + 1
    n_test_classes = round(n_test_datasets ** 0.5) + n_valid_classes + 1

    indices = np.arange(n_test_classes)
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = (indices[:n_train_classes], indices[n_train_classes:n_valid_classes],
                                      indices[n_valid_classes:])
    idxs = [train_idx, valid_idx, test_idx]
    n_datasets = [n_train_datasets, n_train_datasets + n_valid_datasets, n_train_datasets + n_valid_datasets + n_test_datasets]
    binary_dataset_idx = 0
    for index in [0, 1, 2]:
        for first_class in idxs[index]:
            for second_class in idxs[index]:
                if first_class == second_class:
                    continue

                idx = np.arange(len(dataset))
                np.random.shuffle(idx)
                dataset = dataset[idx]
                first_class_filter = dataset[:, target_starting_idx] == first_class
                first_class_x = dataset[first_class_filter, :target_starting_idx]

                second_class_filter = dataset[:, target_starting_idx] == second_class
                second_class_x = dataset[second_class_filter, :target_starting_idx]

                x = torch.vstack((first_class_x[:n_instance_per_class_per_dataset],
                                  second_class_x[:n_instance_per_class_per_dataset]))
                y = torch.ones((config["n_instances_per_dataset"], 1))
                y[:n_instance_per_class_per_dataset] -= 2

                binary_dataset = torch.hstack((x, y))
                if config["shuffle_each_dataset_samples"]:
                    random_indices = torch.randperm(config["n_instances_per_dataset"])
                    binary_dataset = binary_dataset[random_indices]

                binary_datasets[binary_dataset_idx] = binary_dataset
                binary_dataset_idx += 1
                if binary_dataset_idx == n_datasets[index]:
                    break
            if binary_dataset_idx == n_datasets[index]:
                break

    return binary_datasets


def obtain_cifar100_dataset(config: dict) -> torch.Tensor:
    train_set = create_train_set(config)
    test_set = create_test_set(config)

    return torch.vstack((train_set, test_set))


def create_train_set(config: dict) -> torch.Tensor:
    train_set = torchvision.datasets.CIFAR100(root=str(CIFAR100_BASE_PATH), train=True, download=True)
    train_set = apply_transforms_to_dataset(config, train_set)
    n_instances_in_cifar100_train_set = train_set.data.shape[0]

    return torch.hstack((train_set.data.reshape((n_instances_in_cifar100_train_set, config["n_features"])),
                         torch.tensor(train_set.targets).reshape(n_instances_in_cifar100_train_set, -1)))


def create_test_set(config: dict) -> torch.Tensor:
    test_set = torchvision.datasets.CIFAR100(root=str(CIFAR100_BASE_PATH), train=False, download=True)
    test_set = apply_transforms_to_dataset(config, test_set)
    n_instances_in_cifar100_test_set = test_set.data.shape[0]

    return torch.hstack((test_set.data.reshape((n_instances_in_cifar100_test_set, config["n_features"])),
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
