from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

DATA_BASE_PATH = Path("dataset")
MNIST_DEFAULT_IMG_SIZE = (28, 28)
NUMPY_FILE_EXTENSION = ".npy"
MNIST_BASE_PATH = DATA_BASE_PATH / "MNIST"

def load_mnist_binary(config: dict, first_digit: str, second_digit: str) -> np.ndarray:
    expected_datasets_cache_path = create_datasets_cache_path(config, first_digit, second_digit)

    if expected_datasets_cache_path.exists():
        return np.load(expected_datasets_cache_path)

    return create_and_store_mnist_datasets(config, first_digit, second_digit)


def create_datasets_cache_path(config: dict, first_digit, second_digit) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        file_name.append(f"{key}={config[key]}")
    first_second_digits = first_digit + second_digit
    return MNIST_BASE_PATH / first_second_digits / "cache" / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def create_and_store_mnist_datasets(config: dict, first_digit, second_digit) -> np.ndarray:
    train_dataset = create_train_set(config, first_digit, second_digit)
    test_dataset = create_test_set(config, first_digit, second_digit)
    datasets = create_mnist_binary_datasets(config, train_dataset, test_dataset)
    store_mnist_datasets(config, first_digit, second_digit, datasets)

    return datasets


def store_mnist_datasets(config: dict, first_digit, second_digit, datasets: np.ndarray) -> None:
    first_second_digits = first_digit + second_digit
    MNIST_CACHE_BASE_PATH = MNIST_BASE_PATH / first_second_digits / "cache"
    MNIST_CACHE_BASE_PATH.mkdir(exist_ok=True)
    np.save(create_datasets_cache_path(config, first_digit, second_digit), datasets)


def create_mnist_binary_datasets(config: dict, train_dataset, test_dataset) -> np.ndarray:
    binary_datasets = np.zeros((config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] +
                                config["target_size"]))
    n_partition = len(train_dataset) // config["n_instances_per_dataset"]
    num_swap = 0
    stop = False
    while not stop:
        idx = np.arange(len(train_dataset))
        np.random.shuffle(idx)
        train_dataset_shuffled = train_dataset[idx].clone()
        for pixel in range(config["n_pixels_to_permute"]):
            first_pixel_location = np.random.randint(0, config["n_features"])
            second_pixel_location = np.random.randint(0, config["n_features"])
            first_pixel = train_dataset_shuffled[:, first_pixel_location].clone()
            second_pixel = train_dataset_shuffled[:, second_pixel_location].clone()
            train_dataset_shuffled[:, second_pixel_location] = first_pixel
            train_dataset_shuffled[:, first_pixel_location] = second_pixel
        for num_partition in range(n_partition):
            if n_partition * num_swap + num_partition == int(config["n_dataset"] * (config["splits"][0] + config["splits"][1])):
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

        for pixel in range(config["n_pixels_to_permute"]):
            first_pixel_location = np.random.randint(0, config["n_features"])
            second_pixel_location = np.random.randint(0, config["n_features"])
            first_pixel = test_dataset_reduced[:, first_pixel_location].clone()
            second_pixel = test_dataset_reduced[:, second_pixel_location].clone()
            test_dataset_reduced[:, second_pixel_location] = first_pixel
            test_dataset_reduced[:, first_pixel_location] = second_pixel
        binary_datasets[binary_dataset_idx] = test_dataset_reduced
    return binary_datasets


def obtain_mnist_dataset(config: dict, first_digit: str, second_digit: str) -> torch.Tensor:
    train_set = create_train_set(config, first_digit, second_digit)
    test_set = create_test_set(config, first_digit, second_digit)

    return torch.vstack((train_set, test_set))

def create_train_set(config: dict, first_digit, second_digit) -> torch.Tensor:
    assert int(config["n_features"] ** 0.5) == config["n_features"] ** 0.5
    reshaped_size = int(config["n_features"] ** 0.5)
    transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]
    train_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=True, download=True, transform=transforms.Compose(transform))
    n_instances_in_mnist_train_set = train_set.data.shape[0]
    train_feature = torchvision.transforms.functional.resize(train_set.data, (reshaped_size, reshaped_size))
    data = train_feature.data.reshape((n_instances_in_mnist_train_set, config["n_features"]))
    target = train_set.targets.reshape(n_instances_in_mnist_train_set)
    first_indx = target == int(first_digit)
    second_indx = target == int(second_digit)
    target_first = torch.ones((sum(first_indx), 1)) * -1
    target_second = torch.ones((sum(second_indx), 1))
    data_first = data[first_indx]
    data_second = data[second_indx]
    complete_data = torch.hstack((torch.vstack((data_first, data_second)), torch.vstack((target_first, target_second))))

    return complete_data[torch.randperm(complete_data.size()[0])]


def create_test_set(config: dict, first_digit, second_digit) -> torch.Tensor:
    assert int(config["n_features"] ** 0.5) == config["n_features"] ** 0.5
    reshaped_size = int(config["n_features"] ** 0.5)
    transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]
    test_set = torchvision.datasets.MNIST(root=str(MNIST_BASE_PATH), train=False, download=True, transform=transforms.Compose(transform))
    n_instances_in_mnist_test_set = test_set.data.shape[0]
    test_feature = torchvision.transforms.functional.resize(test_set.data, (reshaped_size, reshaped_size))
    data = test_feature.data.reshape((n_instances_in_mnist_test_set, config["n_features"]))
    target = test_set.targets.reshape(n_instances_in_mnist_test_set)
    first_indx = target == int(first_digit)
    second_indx = target == int(second_digit)
    target_first = torch.ones((sum(first_indx), 1)) * -1
    target_second = torch.ones((sum(second_indx), 1))
    data_first = data[first_indx]
    data_second = data[second_indx]
    complete_data = torch.hstack((torch.vstack((data_first, data_second)), torch.vstack((target_first, target_second))))

    return complete_data[torch.randperm(complete_data.size()[0])]
