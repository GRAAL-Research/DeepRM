import math

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def train_valid_loaders(dataset, batch_size, splits, shuffle=True, seed=42):
    """
    Divides a dataset into a training set and a validation set, both in a Pytorch DataLoader form.
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Desired batch-size for the DataLoader
        splits (list of float): Desired proportion of training, validation and test example must sum to 1).
        shuffle (bool): Whether the examples are shuffled before train/validation split.
        seed (int): A random seed.
    Returns:
        Tuple (training DataLoader, validation DataLoader).
    """
    assert sum(splits) == 1, 'The sum of splits must be 1.'
    num_data = len(dataset)
    indices = np.arange(num_data)
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
    train_var = np.var((dataset[indices[:split_1], int(dataset.shape[1]/2):, -1] + 1) / 2)
    valid_var = np.var((dataset[indices[split_1:split_2], int(dataset.shape[1]/2):, -1] + 1) / 2)
    test_var = np.var((dataset[indices[split_2:], int(dataset.shape[1]/2):, -1] + 1) / 2)
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, train_var, valid_var, test_var
