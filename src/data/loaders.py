import math
import itertools as it
from os.path import split

import numpy as np
from torch.utils.data import Subset, DataLoader


def train_valid_and_test_indices(dataset, datasets: np.ndarray, splits: list[float], are_test_classes_shared_with_train: bool,
                                 seed: int, is_shuffling=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert sum(splits) == 1, "The sum of splits must be 1."

    n_datasets = len(datasets)
    datasets_indices = np.arange(n_datasets)
    if not are_test_classes_shared_with_train and dataset == "mnist":
        num_classes = int((1 + math.sqrt(1 + 4 * int(n_datasets))) / 2)
        test_idx = []
        current_class = 0
        while len(test_idx) / n_datasets < splits[2]:
            test_idx += extract_class(num_classes, current_class)
            current_class += 1
        other_idx = []
        for idx in datasets_indices:
            if idx not in test_idx:
                other_idx.append(idx)
        if is_shuffling:
            np.random.seed(seed)
            np.random.shuffle(other_idx)
        split_1 = math.floor(splits[0] / (splits[0] + splits[1]) * len(other_idx))
        train_idx = other_idx[:split_1]
        valid_idx = other_idx[split_1:]

        return np.array(train_idx), np.array(valid_idx), np.array(test_idx)

    if not are_test_classes_shared_with_train and dataset == "cifar100":
        np.random.seed(0)
        used_idx = np.array(range(len(datasets)))
        np.random.shuffle(used_idx)
        split_1 = math.floor(splits[0] / (splits[0] + splits[1]) * 100)
        return np.array(used_idx[:split_1]), np.array(used_idx[split_1:100]), np.array(used_idx[100:150])

    if is_shuffling and dataset != "mnist_multi":
        np.random.seed(seed)
        np.random.shuffle(datasets_indices)

    if n_datasets > 3:
        split_1 = math.floor(splits[0] * n_datasets)
        split_2 = math.floor(splits[1] * n_datasets) + split_1
        train_idx = datasets_indices[:split_1]
        valid_idx = datasets_indices[split_1:split_2]
        test_idx = datasets_indices[split_2:]
        return train_idx, valid_idx, test_idx

    split_1 = math.floor(splits[0] / (1 - splits[2]) * n_datasets)
    train_idx = datasets_indices[:split_1]
    valid_idx = datasets_indices[split_1:]
    test_idx = np.arange(len(datasets[1]))

    return train_idx, valid_idx, test_idx


def create_data_loader(dataset: str, type: str, datasets: np.ndarray, batch_size: int, indices: np.ndarray) -> DataLoader:
    if dataset == 'cifar100':
        if type in ['train', 'valid']:
            datasets = datasets[:, list(range(0, 500)) + list(range(600, 1100))]
            idx = list(range(1000))
        elif type == 'test':
            datasets = datasets[:, list(range(500, 600)) + list(range(1100, 1200))]
            idx = list(range(200))
        for i in range(len(datasets)):
            np.random.shuffle(idx)
            datasets[i] = datasets[i, idx]
    subset = Subset(datasets, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def compute_variances(datasets: np.ndarray, train_idx,
                      valid_idx, test_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_var = np.var((datasets[train_idx, int(datasets.shape[1] / 2):, -1] + 1) / 2)
    valid_var = np.var((datasets[valid_idx, int(datasets.shape[1] / 2):, -1] + 1) / 2)
    test_var = np.var((datasets[test_idx, int(datasets.shape[1] / 2):, -1] + 1) / 2)

    return train_var, valid_var, test_var


def extract_class(num_classes: int, current_class: int) -> list:
    starting_point = num_classes * current_class
    data_number = list(np.arange(starting_point, starting_point + num_classes - current_class - 1))
    for i in range(1, num_classes - current_class):
        data_number.append(starting_point + (num_classes - 1) * i)
    return data_number