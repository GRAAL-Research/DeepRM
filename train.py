import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import copy

class SimpleNet(nn.Module):
    def __init__(self, input_dim, kernel_dim, hidden_dim, output_dim, m, d, batch_size):
        """
        Generates a simple feed-forward neural network with ReLU activations and kernel mean embedding (MLP).
            The kernel has |kernel_dim| layer; the element outputted by the Kernel are multiplied by the label values
            before applying the mean function. Then, there are |hidden_dim| hidden layers, and an output layer.
        Args:
            intput_dim (int): Input dimension of the network.
            kernel_dim (list of int): Dimensions of the kernel mean embedding - composed of |kernel_dim| hidden layers.
            hidden_dim (list of int): Dimensions of the hidden layers. There are |hidden_dim| hidden layers.
            output_dim (int): Outputput dimension of the network
            m (int): Number of examples in each class, per dataset
            d (int): Input dimension of each dataset
            batch_size (int): Batch size.
        Output:
            nn.Module A feedforward neural network.
        """
        super(SimpleNet, self).__init__()
        self.m, self.d, self.k, self.batch_size = m, d, kernel_dim[-1], batch_size
        self.kernel = torch.nn.ModuleList()

        ker_dims = [input_dim] + kernel_dim
        for i in range(len(ker_dims)-1):
            self.kernel.append(nn.Linear(ker_dims[i], ker_dims[i+1]))
            if i < len(ker_dims) - 2:
                self.kernel.append(nn.ReLU())

        self.hidden = torch.nn.ModuleList()
        hid_dims = [kernel_dim[-1]] + hidden_dim + [output_dim]
        for i in range(len(hid_dims)-1):
            self.hidden.append(nn.Linear(hid_dims[i], hid_dims[i+1]))
            if i < len(hid_dims) - 2:
                self.hidden.append(nn.ReLU())

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input
        return:
            torch.tensor Output of the network
        """
        x_t = torch.zeros((self.batch_size, self.k))
        for i in range(len(x)):
            x_i = x[i][:,:-1]
            for layer in self.kernel:
                x_i = layer(x_i)
            x_t[i] = torch.mean(x_i * torch.reshape(x[i][:,-1], (-1,1)), dim=0)
        for layer in self.hidden:
            x_t = layer(x_t)
        return x_t

def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True, seed=42):
    """
    Divides a dataset into a training set and a validation set, both in a Pytorch DataLoader form.
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Desired batch-size for the DataLoader
        train_split (float): Desired proportion of training example.
        shuffle (bool): Whether the examples are shuffled before train/validation split.
        seed (int): A random seed.
    Returns:
        Tuple (training DataLoader, validation DataLoader).
    """
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader

def train(meta_pred, pred, dataset, train_split, optimizer, scheduler, tol, early_stop, n_epoch, batch_size, criterion, DEVICE):
    """
    Trains a meta predictor using PyTorch.

    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (function): A predictor whose parameters are computed by the meta predictor.
        dataset (Dataset): A dataset.
        train_split (float): Desired proportion of training example.
        optimizer (torch.optim): Optimisation algorithm
        scheduler (torch.optim.lr_scheduler): Scheduler for the learning rate
        tol (float): Quantity by which the loss must diminishes in order for this increment not to be marginal
        early_stop (int): Number of epochs by which, if the loss hasn't diminished by tol (see above), the training stops
        n_epoch (int): The maximum number of epochs.
        batch_size (int): Batch size.
        criterion (torch.nn, function): Loss function
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu

    Returns:
        Tuple (nn.Module, float, float): The trained model; the best train (and validation) accuracy
    """
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size, train_split)
    best_val_acc, best_train_acc, j, hist = 0, 0, 0, {'epoch': [],
                                                      'train_loss': [],
                                                      'valid_loss': [],
                                                      'train_acc': [],
                                                      'valid_acc': []}
    for i in range(n_epoch):
        meta_pred.train()
        with torch.enable_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                if str(DEVICE) == 'cuda':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                meta_output = meta_pred(inputs)
                output = pred(meta_output, inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
        train_acc, train_loss = perf(meta_pred, pred, criterion, train_loader, DEVICE)
        val_acc, val_loss = perf(meta_pred, pred, criterion, valid_loader, DEVICE)
        update_hist(hist, (train_acc, train_loss, val_acc, val_loss, i))
        #history.save(dict(acc=train_acc, val_acc=val_acc, loss=train_loss, val_loss=val_loss, lr=optimizer.state_dict()['param_groups'][0]['lr']))
        print(f'Epoch {i + 1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f}')
        scheduler.step(val_acc)
        if i == 1 or val_acc > best_val_acc + tol:
            j = copy.copy(i)
        #    cop = copy.deepcopy(meta_pred)
        #    best_val_acc = val_acc
        #    best_train_acc = train_acc
        if i - j > early_stop:
            break
    return hist

def lin_clas(weights, inputs, return_sign = False):
    """
    Generates the predicton of a (or many) linear classifier, given its (their) weights and inputs.
    Args:
        weights (torch.tensor of size batch_size x (d+1)): The d first columns represent the orientation of the linear
            predictors, the last element is their bias.
        inputs (torch.tensor of size batch_size x m x d): The features of examples in a batch
        return_sign (bool): Whether to return only the output with sigmoid function, or also with the sign function applied.
    Returns:
        Torch.tensor of size batch_size x m (or tuple of two torch.tens...), the output with sigmoid (and sign) applied.
    """
    out = torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1) * weights[:, :-1], dim=-1) + weights[:, -1]
    out = torch.transpose(out, 0, 1)
    if not return_sign:
        return torch.sigmoid(out)
    return torch.sigmoid(out), torch.sign(out)

def update_hist(hist, values):
    """
    Adds values to the hist dictionnary to keep track of the losses and accuracies for each epochs.
    Args:
        hist (dic): A dictionnary that keep track of training metrics.
        values (Tuple): Elements to be added to the dictionnary.
    """
    hist['epoch'].append(values[0])
    hist['train_acc'].append(values[1])
    hist['train_loss'].append(values[2])
    hist['valid_acc'].append(values[3])
    hist['valid_loss'].append(values[4])

def lin_loss(output, targets):
    """
    Computes the total linear loss.
    Args:
        output (torch.tensor of size batch_size x m): The output (0 or 1) of the predictor
        targets (torch.tensor of size batch_size x m): The labels (0 or 1)
    Return:
        Float, the total linear loss incurred.
    """
    return torch.sum(((output * (targets * 2 - 1)) + 1) / 2)

def perf(meta_pred, pred, criterion, data_loader, DEVICE):
    """
    Computes the overall accuracy and loss of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (function): A predictor whose parameters are computed by the meta predictor.
        criterion (torch.nn, function): Loss function
        data_loader (DataLoader): A DataLoader to test on.
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        tot_loss = 0
        tot_acc = 0
        for inputs, targets in data_loader:
            i += len(targets) * len(targets[0])
            inputs, targets = inputs.float(), targets.float()
            if str(DEVICE) == 'cuda':
                inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
            meta_output = meta_pred(inputs)
            output = pred(meta_output, inputs, return_sign = True)
            tot_loss += criterion(output[0], targets)
            tot_acc += lin_loss(output[1], targets)
        return tot_acc / i, tot_loss / i