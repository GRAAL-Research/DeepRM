from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

DATA_BASE_PATH = Path("dataset")
MNIST_BASE_PATH = DATA_BASE_PATH / "MNIST"
MNIST_DEFAULT_IMG_SIZE = (28, 28)
MNIST_CACHE_BASE_PATH = MNIST_BASE_PATH / "cache"
NUMPY_FILE_EXTENSION = ".npy"


def load_mnist_label(config: dict) -> np.ndarray:
    expected_datasets_cache_path = create_datasets_cache_path(config)

    if expected_datasets_cache_path.exists():
        return np.load(expected_datasets_cache_path)

    return create_and_store_mnist_datasets(config)


def create_datasets_cache_path(config: dict) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        file_name.append(f"{key}={config[key]}")

    return MNIST_CACHE_BASE_PATH / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def create_and_store_mnist_datasets(config: dict) -> np.ndarray:
    train_dataset = create_train_set(config)
    test_dataset = create_test_set(config)
    datasets = create_mnist_binary_datasets(config, train_dataset, test_dataset)
    store_mnist_datasets(config, datasets)

    return datasets


def store_mnist_datasets(config: dict, datasets: np.ndarray) -> None:
    MNIST_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config), datasets)


def create_mnist_binary_datasets(config: dict, train_dataset, test_dataset) -> np.ndarray:
    binary_datasets = np.zeros((config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] +
                                config["target_size"]))
    train_dataset = torch.hstack((train_dataset[:, :-1], F.one_hot(train_dataset[:, -1]) * 2 - 1))
    test_dataset = torch.hstack((test_dataset[:, :-1], F.one_hot(test_dataset[:, -1]) * 2 - 1))
    n_partition = len(train_dataset) // config["n_instances_per_dataset"]
    num_swap = 0
    stop = False
    while not stop:
        idx = np.arange(len(train_dataset))
        np.random.shuffle(idx)
        train_dataset_shuffled = train_dataset[idx].clone()
        label_idx = np.arange(784, 794)
        np.random.shuffle(label_idx)
        train_dataset_shuffled[:, -10:] = train_dataset_shuffled[:, label_idx]
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
        idx = np.arange(len(test_dataset))
        np.random.shuffle(idx)
        test_dataset_reduced = test_dataset[idx[:config["n_instances_per_dataset"]]]

        label_idx = np.arange(784, 794)
        np.random.shuffle(label_idx)
        test_dataset_reduced[:, -10:] = test_dataset_reduced[:, label_idx]
        binary_datasets[binary_dataset_idx] = test_dataset_reduced
    return binary_datasets


def obtain_mnist_dataset(config: dict) -> torch.Tensor:
    train_set = create_train_set(config)
    test_set = create_test_set(config)

    return torch.vstack((train_set, test_set))


def create_train_set(config: dict) -> torch.Tensor:
    transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]
    train_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=True, download=True,
                                           transform=transforms.Compose(transform))
    n_instances_in_mnist_train_set = train_set.data.shape[0]
    data = train_set.data.reshape((n_instances_in_mnist_train_set, config["n_features"]))
    target = train_set.targets.reshape(n_instances_in_mnist_train_set, -1)

    return torch.hstack((data, target))


def create_test_set(config: dict) -> torch.Tensor:
    transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]
    test_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=False, download=True,
                                          transform=transforms.Compose(transform))
    n_instances_in_mnist_test_set = test_set.data.shape[0]

    return torch.hstack((test_set.data.reshape((n_instances_in_mnist_test_set, config["n_features"])),
                         test_set.targets.reshape(n_instances_in_mnist_test_set, -1)))
