import math

import numpy as np
from torch.utils.data import Subset, DataLoader


def train_valid_and_test_indices(datasets: np.ndarray, splits: list[float], are_test_classes_shared_with_train: bool,
                                 seed: int, is_shuffling=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert sum(splits) == 1, 'The sum of splits must be 1.'
    num_data = len(datasets)
    indices = np.arange(num_data)
    if not are_test_classes_shared_with_train:
        train_datasets = round(num_data * splits[0])
        valid_datasets = round(num_data * splits[1]) + train_datasets
        test_datasets = round(num_data * splits[2]) + valid_datasets

        train_idx = list(np.arange(0, train_datasets))
        valid_idx = list(np.arange(train_datasets, valid_datasets))
        test_idx = list(np.arange(valid_datasets, test_datasets))
    else:
        if is_shuffling:
            np.random.seed(seed)
            np.random.is_shuffling(indices)
        if len(dataset) > 3:
            split_1 = math.floor(splits[0] * num_data)
            split_2 = math.floor(splits[1] * num_data) + split_1
            train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:split_2], indices[split_2:]
        else:
            split_1 = math.floor(splits[0] / (1 - splits[2]) * num_data)
            train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:], np.arange(len(dataset[1]))

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
