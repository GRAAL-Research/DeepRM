import math
import torch

import numpy as np
from torch.utils.data import Subset, DataLoader


def train_valid_and_test_indices(dataset, datasets: np.ndarray, splits: list[float], are_test_classes_shared_with_train: bool,
                                 seed: int, is_shuffling=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert sum(splits) == 1, "The sum of splits must be 1."

    n_datasets = len(datasets)
    datasets_indices = np.arange(n_datasets)
    if not are_test_classes_shared_with_train and dataset == "mnist":
        num_classes = int((1 + math.sqrt(1 + 4 * int(n_datasets))) / 2)
        valid_idx = []
        test_idx = []
        current_class = 0
        while len(test_idx) / n_datasets < splits[2]:
            test_idx += extract_class(num_classes, current_class)
            current_class += 1
        valid_idx += extract_class(num_classes, current_class)
        other_idx = []
        for idx in datasets_indices:
            if idx not in valid_idx:
                if idx not in test_idx:
                    other_idx.append(idx)
        if is_shuffling:
            np.random.seed(seed)
            np.random.shuffle(other_idx)
        train_idx = other_idx
        return np.array(train_idx), np.array(valid_idx), np.array(test_idx)

    if not are_test_classes_shared_with_train and dataset == "cifar100":
        used_idx = np.array(range(100))
        np.random.shuffle(used_idx)
        split_1 = math.floor(splits[0] / (splits[0] + splits[1]) * 100)
        return np.array(used_idx[:split_1]), np.array(used_idx[split_1:]), np.array(range(100, 150))

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


def create_data_loader(datasets: np.ndarray, meta_batch_size: int, indices: np.ndarray) -> DataLoader:
    subset = Subset(datasets, indices)
    return DataLoader(subset, batch_size=meta_batch_size, shuffle=True, collate_fn=collate_fn_padd)


def compute_variances(datasets: np.ndarray, train_idx,
                      valid_idx, test_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_labels, valid_labels, test_labels = [], [], []
    if type(datasets[0]) is np.ndarray:
        datasets = torch.tensor(datasets)
    for i in range(len(datasets)):
        if i in train_idx:
            train_labels.append(datasets[i][:, -1])
        elif i in valid_idx:
            valid_labels.append(datasets[i][:, -1])
        elif i in test_idx:
            test_labels.append(datasets[i][:, -1])
    train_var = torch.var(torch.hstack(train_labels)).item()
    valid_var = torch.var(torch.hstack(valid_labels)).item()
    test_var = torch.var(torch.hstack(test_labels)).item()

    return train_var, valid_var, test_var


def extract_class(num_classes: int, current_class: int) -> list:
    starting_point = num_classes * current_class
    data_number = list(np.arange(starting_point, starting_point + num_classes - current_class - 1))
    for i in range(1, num_classes - current_class):
        data_number.append(starting_point + (num_classes - 1) * i)
    return data_number


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ])
    ## padd
    batch = [ torch.Tensor(t) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0)
    return torch.transpose(batch, 0, 1), lengths, mask
