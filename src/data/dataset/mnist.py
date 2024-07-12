from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms

MNIST_DATASET_PATH = Path("dataset")


def load_mnist(config: dict) -> np.ndarray:
    assert config["n_dataset"] <= 10 * (10 - 1), "We can't create more than 90 MNIST binary datasets."
    assert config["n_instances_per_dataset"] % 2 == 0, "The number of instances per dataset must be even."
    assert config["n_features"] == 28 * 28, "The number of features must be equivalent to 28x28 pixels."

    dataset = obtain_mnist_dataset(config)
    return create_mnist_binary_datasets(config, dataset)


def create_mnist_binary_datasets(config, dataset):
    target_dim = 1
    binary_datasets = np.zeros(
        (config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] + target_dim))

    n_instance_per_class_per_dataset = config["n_instances_per_dataset"] // 2
    binary_dataset_idx = 0
    n_digits = 10
    target_idx = -1

    for first_class in range(n_digits):
        for second_class in range(n_digits):
            if binary_dataset_idx == config["n_dataset"]:
                break
            if first_class == second_class:
                continue

            first_class_filter = dataset[:, target_idx] == first_class
            first_class_x = dataset[first_class_filter, :target_idx]

            second_class_filter = dataset[:, target_idx] == second_class
            second_class_x = dataset[second_class_filter, :target_idx]

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

        if binary_dataset_idx == config["n_dataset"]:
            break

    return binary_datasets


def obtain_mnist_dataset(config: dict) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = create_train_set(config, transform)
    test_set = create_test_set(config, transform)

    return torch.vstack((train_set, test_set))


def create_train_set(config: dict, transform) -> torch.Tensor:
    train_set = torchvision.datasets.MNIST(root=str(MNIST_DATASET_PATH), train=True, download=True, transform=transform)
    n_instances_in_mnist_train_set = train_set.data.shape[0]
    data = train_set.data.reshape((n_instances_in_mnist_train_set, config["n_features"]))
    target = train_set.targets.reshape(n_instances_in_mnist_train_set, -1)

    return torch.hstack((data, target))


def create_test_set(config: dict, transform) -> torch.Tensor:
    test_set = torchvision.datasets.MNIST(root=str(MNIST_DATASET_PATH), train=False, download=True, transform=transform)
    n_instances_in_mnist_test_set = test_set.data.shape[0]
    return torch.hstack((test_set.data.reshape((n_instances_in_mnist_test_set, config["n_features"])),
                         test_set.targets.reshape(n_instances_in_mnist_test_set, -1)))
