import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import copy

class SimpleNet(nn.Module):
    def __init__(self, input_dim, kernel_dim, hidden_dim, output_dim, m, d, batch_size):
        """
        Generates a simple feed-forward neural network with ReLU activations, with kernel mean embedding (MLP).
        Args:
            intput_dim (int): Input dimension of the network.
            kernel_dim (list of int): Dimensions of the kernel mean embedding - composed of |kernel_dim| hidden layers.
            hidden_dim (list of int): Dimensions of the hidden layers. There are |hidden_dim| hidden layers.
            output_dim (int): Outputput dimension of the network
        Output:
            nn.Module A network.
        """
        super(SimpleNet, self).__init__()
        self.m, self.d, self.k, self.batch_size = m, d, hidden_dim[0], batch_size
        self.kernel = torch.nn.ModuleList()

        ker_dims = [input_dim] + kernel_dim + [hidden_dim[0]]
        for i in range(len(ker_dims)-1):
            self.kernel.append(nn.Linear(ker_dims[i], ker_dims[i+1]))
            if i < len(ker_dims) - 2:
                self.kernel.append(nn.ReLU())

        self.hidden = torch.nn.ModuleList()
        hid_dims = hidden_dim + [output_dim]
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
        train_split (float): A number between 0 and 1 corresponding to the desired proportion of training example.
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

def train(net, dataset, train_valid, optimizer, scheduler, early_stop, n_epoch, batch_size, criterion, DEVICE):
    """
    Trains a feedforward neural network using PyTorch.

    Args:
        net (nn.Module): A neural network to train.
        dataset (Dataset): A dataset.
        m (int): Number of instances per class in each datasets
        optimizer (torch.optim): Optimisation algorithm
        scheduler (torch.optim.lr_scheduler): Scheduler for the learning rate
        n_epoch (int): The maximum number of epochs.
        batch_size (int): Batch size.
        criterion (torch.nn, function): Loss function
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu

    Returns:
        Tuple (nn.Module, float, float): The trained model; the best train (and validation) accuracy
    """
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size, train_valid)
    best_val_acc, best_train_acc, j, hist = 0, 0, 0, {'epoch':[],
                                                      'train':[],
                                                      'valid':[]}
    for i in range(n_epoch):
        net.train()
        with torch.enable_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                if str(DEVICE) == 'cuda':
                    inputs, targets, net = inputs.cuda(), targets.cuda(), net.cuda()
                output = net(inputs)
                loss = criterion(output, inputs, targets)
                loss.backward()
                optimizer.step()
        train_acc = perf(net, train_loader, DEVICE)
        val_acc = perf(net, valid_loader, DEVICE)
        hist['epoch'].append(i)
        hist['train'].append(train_acc)
        hist['valid'].append(val_acc)
        #history.save(dict(acc=train_acc, val_acc=val_acc, loss=train_loss, val_loss=val_loss, lr=optimizer.state_dict()['param_groups'][0]['lr']))
        print(f'Epoch {i + 1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f}')
        scheduler.step(val_acc)
        if i == 1 or val_acc > best_val_acc:
            cop = copy.deepcopy(net)
            best_val_acc = val_acc
            best_train_acc = train_acc
            j = copy.copy(i)
        if i - j > early_stop:
            break
    return hist#cop, best_train_acc, best_val_acc

def BCELoss_mod(output, inputs, targets):
    """
    Computes the Binary Cross Entropy Loss, given features (inputs), there labels (targets) and a linear predictor.
    Args:
        output (torch.tensor of size batch_size x (d+1)): The d first columns represent the orientation of the linear
            predictor, the last element is its bias.
        inputs (torch.tensor of size batch_size x m x d): The features of examples in a batch
        targets (torch.tensor of size batch_size x m): The targets.
    Returns:
        Float the loss incured
    """
    out = nn.Sigmoid()(torch.sum(torch.transpose(inputs[:,:,:-1], 0, 1) * output[:, :-1], dim=-1) + output[:,-1])
    return nn.BCELoss()(torch.transpose(out,0,1), targets)

def perf(net, data_loader, DEVICE):
    """
    Computes the overall performance of a predictor on a given task.
    Args:
        net (nn.Module): A neural network.
        data_loader (DataLoader): A DataLoader to test on.
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu

    Returns:
        float: the mean 0-1 accuracy.
    """
    net.eval()
    with torch.no_grad():
        i = 0
        tot_loss = 0
        for inputs, targets in data_loader:
            i += len(targets) * len(targets[0])
            inputs, targets = inputs.float(), targets.float()
            if str(DEVICE) == 'cuda':
                inputs, targets, net = inputs.cuda(), targets.cuda(), net.cuda()
            output = net(inputs)
            out = torch.transpose(torch.sign(torch.sum(torch.transpose(inputs[:,:,:-1], 0, 1) * output[:, :-1], dim=-1) + output[:,-1]), 0, 1)
            tot_loss += torch.sum(((out*(targets*2-1))+1)/2)
        return tot_loss / i