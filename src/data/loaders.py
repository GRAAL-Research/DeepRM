import math

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def train_valid_loaders(dataset, batch_size, splits, are_test_classes_shared_with_train, shuffle=True, seed=42) -> (
        tuple)[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Divides a dataset into a training set and a validation set, both in a Pytorch DataLoader form.
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Desired batch-size for the DataLoader
        splits (list of float): Desired proportion of training, validation and test example must sum to 1).
        are_test_classes_shared_with_train ():
        shuffle (bool): Whether the examples are shuffled before train/validation split.
        seed (int): A random seed.
    Returns:
        Tuple (training DataLoader, validation DataLoader, test DataLoader, float, float, float, list).
    """
    assert sum(splits) == 1, 'The sum of splits must be 1.'
    num_data = len(dataset)
    indices = np.arange(num_data)
    if not are_test_classes_shared_with_train:
        num_classes = int((1 + math.sqrt(1 + 4 * int(num_data))) / 2)
        test_idx = []
        curr_class = -1
        while len(test_idx) / num_data < splits[2]:
            curr_class += 1
            test_idx += extract_class(num_classes, curr_class)
        other_idx = []
        for idx in indices:
            if idx not in test_idx:
                other_idx.append(idx)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(other_idx)
        split_1 = math.floor(splits[0] / (splits[0] + splits[1]) * len(other_idx))
        train_idx, valid_idx = other_idx[:split_1], other_idx[split_1:]
    else:
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        if len(dataset) > 3:
            split_1 = math.floor(splits[0] * num_data)
            split_2 = math.floor(splits[1] * num_data) + split_1
            train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:split_2], indices[split_2:]
        else:
            split_1 = math.floor(splits[0] / (1 - splits[2]) * num_data)
            train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:], np.arange(len(dataset[1]))

    train_var = np.var((dataset[train_idx, int(dataset.shape[1] / 2):, -1] + 1) / 2)
    valid_var = np.var((dataset[valid_idx, int(dataset.shape[1] / 2):, -1] + 1) / 2)
    test_var = np.var((dataset[test_idx, int(dataset.shape[1] / 2):, -1] + 1) / 2)
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    idx = [train_idx, valid_idx, test_idx]

    return train_loader, valid_loader, test_loader, train_var, valid_var, test_var, idx


def extract_class(num_classes, curr_class):
    starting_point = num_classes * curr_class
    data_number = list(np.arange(starting_point, starting_point + num_classes - curr_class - 1))
    for i in range(1, num_classes - curr_class):
        data_number.append(starting_point + (num_classes - 1) * i)
    return data_number
