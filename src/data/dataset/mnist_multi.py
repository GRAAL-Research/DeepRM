import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

DATA_BASE_PATH = Path("dataset")
MNIST_BASE_PATH = DATA_BASE_PATH / "MNIST"
MNIST_DEFAULT_IMG_SIZE = (28, 28)
MNIST_CACHE_BASE_PATH = MNIST_BASE_PATH / "cache"
NUMPY_FILE_EXTENSION = ".npy"


def load_mnist_multi(config: dict) -> np.ndarray:
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
    dataset = create_train_set(config)
    datasets = create_mnist_binary_datasets(config, dataset)
    store_mnist_datasets(config, datasets)

    return datasets


def store_mnist_datasets(config: dict, datasets: np.ndarray) -> None:
    MNIST_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config), datasets)


def create_mnist_binary_datasets(config: dict, dataset) -> np.ndarray:
    binary_datasets = np.zeros((config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] +
                                config["target_size"]))
    dataset = dataset[:config["n_instances_per_dataset"]]
    dataset = torch.hstack((dataset[:, :-1], F.one_hot(dataset[:, -1]) * 2 - 1))
    for binary_dataset_idx in range(config["n_dataset"]):
        if binary_dataset_idx == config["n_dataset"]:
            break
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        dataset = dataset[idx]
        dataset_shuffled = dataset.clone()
        for i in range(config["n_pixels_to_permute"]):
            first_pixel_location = np.random.randint(low=0, high=config["n_features"])
            second_pixel_location = np.random.randint(low=0, high=config["n_features"])
            first_pixel = dataset_shuffled[:, first_pixel_location].clone()
            second_pixel = dataset_shuffled[:, second_pixel_location].clone()
            dataset_shuffled[:, second_pixel_location] = first_pixel
            dataset_shuffled[:, first_pixel_location] = second_pixel
        binary_datasets[binary_dataset_idx] = dataset_shuffled
        binary_dataset_idx += 1
    return binary_datasets


def obtain_mnist_dataset(config: dict) -> torch.Tensor:
    train_set = create_train_set(config)
    test_set = create_test_set(config)

    return torch.vstack((train_set, test_set))


def create_train_set(config: dict) -> torch.Tensor:
    train_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=True, download=True)
    train_set = apply_transforms_to_dataset(config, train_set)
    n_instances_in_mnist_train_set = train_set.data.shape[0]
    data = train_set.data.reshape((n_instances_in_mnist_train_set, config["n_features"]))
    target = train_set.targets.reshape(n_instances_in_mnist_train_set, -1)

    return torch.hstack((data, target))


def create_test_set(config: dict) -> torch.Tensor:
    test_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=False, download=True)
    test_set = apply_transforms_to_dataset(config, test_set)
    n_instances_in_mnist_test_set = test_set.data.shape[0]

    return torch.hstack((test_set.data.reshape((n_instances_in_mnist_test_set, config["n_features"])),
                         test_set.targets.reshape(n_instances_in_mnist_test_set, -1)))


def apply_transforms_to_dataset(config: dict, dataset: MNIST) -> MNIST:
    square_root_of_n_features = np.sqrt(config["n_features"])
    is_a_perfect_square = 0 <= config["n_features"] == int(square_root_of_n_features) ** 2
    assert is_a_perfect_square, "The number of features must be a perfect square."

    new_img_size = (int(square_root_of_n_features), int(square_root_of_n_features))

    if new_img_size == MNIST_DEFAULT_IMG_SIZE:
        return dataset

    transform = transforms.Compose([to_pil_image, transforms.Resize(new_img_size), ToTensor()])

    transformed_data = []
    for img in tqdm(dataset.data, desc="Applying MNIST's transforms"):
        img = transform(img)
        transformed_data.append(img)

    dataset.data = torch.stack(transformed_data).squeeze(1)

    return dataset
