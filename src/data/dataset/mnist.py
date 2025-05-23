from pathlib import Path

import math
import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision import transforms as transforms
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH
from src.data.utils import validate_n_features_for_images

DATA_BASE_PATH = Path("dataset")
MNIST_BASE_PATH = DATA_BASE_PATH / "MNIST"
MNIST_DEFAULT_IMG_SIZE = (28, 28)
MNIST_CACHE_BASE_PATH = MNIST_BASE_PATH / "cache"
NUMPY_FILE_EXTENSION = ".npy"


def load_mnist(config: dict) -> np.ndarray:
    expected_datasets_cache_path = create_datasets_cache_path(config)

    if expected_datasets_cache_path.exists():
        return np.load(expected_datasets_cache_path, allow_pickle=True)

    return create_and_store_mnist_datasets(config)


def create_datasets_cache_path(config: dict) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        file_name.append(f"{key}={config[key]}")

    return MNIST_CACHE_BASE_PATH / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def create_and_store_mnist_datasets(config: dict) -> np.ndarray:
    train, test = obtain_mnist_dataset(config)
    datasets = create_mnist_binary_datasets(config, train, test)
    store_mnist_datasets(config, datasets)

    return datasets


def store_mnist_datasets(config: dict, datasets: np.ndarray) -> None:
    MNIST_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config), datasets)


def create_mnist_binary_datasets(config: dict, train, test) -> np.ndarray:
    max_digits = 10
    maximum_of_datasets = max_digits * (max_digits - 1)
    assertion_msg = f"Given {max_digits} digits, we can't create more than {maximum_of_datasets} MNIST binary datasets."
    assert config["n_dataset"] <= maximum_of_datasets, assertion_msg
    n_digits = (1 + math.sqrt(1 + 4 * int(config["n_dataset"]))) / 2
    assert n_digits.is_integer(), f"A round number of classes should be deduced from given number of datasets."
    n_digits = int(n_digits)
    assert config["n_instances_per_dataset"] % 2 == 0, "The number of instances per dataset must be even."

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


def obtain_mnist_dataset(config: dict) -> tuple[Tensor, Tensor]:
    train_set = create_train_set(config)
    test_set = create_test_set(config)

    return train_set, test_set


def create_train_set(config: dict) -> torch.Tensor:
    train_set = torchvision.datasets.MNIST(root=str(DATA_BASE_PATH), train=True, download=True)
    train_set = apply_transforms_to_dataset(config, train_set)
    n_instances_in_mnist_train_set = train_set.data.shape[0]
    data = train_set.data.reshape((n_instances_in_mnist_train_set, config["n_features"]))
    target = train_set.targets.reshape(n_instances_in_mnist_train_set, -1)

    return torch.hstack((data, target))


def create_test_set(config: dict) -> torch.Tensor:
    test_set = torchvision.datasets.MNIST(root=str(DATA_BASE_PATH), train=False, download=True)
    test_set = apply_transforms_to_dataset(config, test_set)
    n_instances_in_mnist_test_set = test_set.data.shape[0]

    return torch.hstack((test_set.data.reshape((n_instances_in_mnist_test_set, config["n_features"])),
                         test_set.targets.reshape(n_instances_in_mnist_test_set, -1)))


def apply_transforms_to_dataset(config: dict, dataset: MNIST) -> MNIST:
    square_root_of_n_features = np.sqrt(config["n_features"])
    is_a_perfect_square = 0 <= config["n_features"] == int(square_root_of_n_features) ** 2
    assert is_a_perfect_square, "The number of features must be a perfect square."
    validate_n_features_for_images(config)

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
