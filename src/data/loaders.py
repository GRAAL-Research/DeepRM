import math
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def train_valid_and_test_indices(dataset_name, datasets: np.ndarray, splits: list[float],
                                 are_test_classes_shared_with_train: bool,
                                 seed: int, is_shuffling=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a few parameters, returns three np.array containing the dataset indices for the train, valid and test meta-
        datasets.
    """
    assert sum(splits) == 1, "The sum of splits must be 1."
    n_datasets = len(datasets)

    assert n_datasets > 3, "There needs to be at least 3 datasets!"
    datasets_indices = np.arange(n_datasets)
    if not are_test_classes_shared_with_train:
        if dataset_name == "mnist_binary":
            # If dataset is mnist_binary, the test meta-dataset will contain tasks with digit 0, then 1, 2, and so on
            # until its proportion is filled.
            n_classes = int((1 + math.sqrt(1 + 4 * int(n_datasets))) / 2)
            valid_idx = []
            test_idx = []
            current_class = 0
            while len(test_idx) / n_datasets < splits[2]:
                test_idx += extract_class(n_classes, current_class)
                current_class += 1
            # Same goes for validation tasks
            while len(valid_idx) / n_datasets < splits[1]:
                valid_idx += extract_class(n_classes, current_class)
                current_class += 1
            other_idx = []
            # The remaining indices corresponds to the train meta-set.
            for idx in datasets_indices:
                if idx not in valid_idx:
                    if idx not in test_idx:
                        other_idx.append(idx)
            if is_shuffling:
                np.random.seed(seed)
                np.random.shuffle(other_idx)
            train_idx = other_idx
            return np.array(train_idx), np.array(valid_idx), np.array(test_idx)

    if is_shuffling:
        np.random.seed(seed)
        np.random.shuffle(datasets_indices)

    # Otherwise, a random partitioning of the datasets is done with proportions respecting "splits".
    split_1 = math.floor(splits[0] * n_datasets)
    split_2 = math.floor(splits[1] * n_datasets) + split_1
    train_idx = datasets_indices[:split_1]
    valid_idx = datasets_indices[split_1:split_2]
    test_idx = datasets_indices[split_2:]
    return train_idx, valid_idx, test_idx


def create_data_loader(datasets: np.ndarray, meta_batch_size: int, indices: np.ndarray) -> DataLoader:
    subset = Subset(datasets, indices)
    return DataLoader(subset, batch_size=meta_batch_size, shuffle=True, collate_fn=collate_fn_padd)


def compute_variances(datasets: np.ndarray, train_idx,
                      valid_idx, test_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A simple function to compute the variance of the labels for the train, valid and test data.
    """
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
    """
    Extracts the dataset idx corresponding to a given class, given the total number of classes.
    """
    starting_point = num_classes * current_class
    data_number = list(np.arange(starting_point, starting_point + num_classes - current_class - 1))
    for i in range(1, num_classes - current_class):
        data_number.append(starting_point + (num_classes - 1) * i)
    return data_number


def collate_fn_padd(batch):
    """
    Pads batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch])
    ## padd
    batch = [torch.Tensor(t) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0)
    return torch.transpose(batch, 0, 1), lengths, mask
