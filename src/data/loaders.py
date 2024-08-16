import math

import numpy as np
from torch.utils.data import Subset, DataLoader


def train_valid_and_test_indices(datasets: np.ndarray, splits: list[float]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    assert sum(splits) == 1, 'The sum of splits must be 1.'
    n_datasets = len(datasets)
    n_train_datasets = math.floor(n_datasets * splits[0])
    n_valid_datasets = math.floor(n_datasets * splits[1]) + n_train_datasets
    n_test_datasets = math.floor(n_datasets * splits[2]) + n_valid_datasets

    train_idx = np.arange(0, n_train_datasets)
    valid_idx = np.arange(n_train_datasets, n_valid_datasets)
    test_idx = np.arange(n_valid_datasets, n_test_datasets)

    return train_idx, valid_idx, test_idx


def create_data_loader(datasets: np.ndarray, batch_size: int, indices: np.ndarray) -> DataLoader:
    subset = Subset(datasets, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


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