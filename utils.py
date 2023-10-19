import numpy as np
import random
from copy import copy
from matplotlib import pyplot as plt
from utils import *
import torch

def lin_loss(output, targets):
    """
    Computes the total linear loss.
    Args:
        output (torch.tensor of size batch_size x m): The output (0 or 1) of the predictor
        targets (torch.tensor of size batch_size x m): The labels (0 or 1)
    Return:
        Float, the total linear loss incurred.
    """
    return torch.mean(((output * (targets * 2 - 1)) + 1) / 2)

def set_seed(seed):
    """
    Sets the seed to a certain value for several packages
    Args:
        seed (int): A seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def l1(x, c):
    """
    Computes the l1 loss, given inputs and a regularization parameter.
    Args:
        x (torch.tensor of size m): Inputs
        targets (c): Regularization parameter
    Return:
        Float, the l1 loss
    """
    return torch.mean(torch.abs(x)) * c

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
    hist['train_acc'].append(values[0])
    hist['train_loss'].append(values[1])
    hist['valid_acc'].append(values[2])
    hist['valid_loss'].append(values[3])
    hist['bound_value'].append(values[4])
    hist['n_sigma'].append(values[5])
    hist['n_Z'].append(values[6])
    hist['epoch'].append(values[7])

def get_lr(optimizer):
    """
        Returns the current lr of an optimizer
        Args:
            optimizer (torch.optim): An optimizer for SGD training.
        Returns:
            Float: the learning rate of the optimizer
        """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show(meta_pred, data_loader, vis, DEVICE):
    """
    Computes the overall accuracy and loss of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        data_loader (DataLoader): A DataLoader to test on.
        vis (int): Number of created predictors to visualize (plot) at the end of the training (only works for
                        linear classifier with 2-d datasets).
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        for inputs, targets in data_loader:
            if i < vis:
                i += 1
                inputs, targets = inputs.float(), targets.float()
                if str(DEVICE) == 'cuda':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                meta_output = meta_pred(inputs)[0]
                inds = inputs[0, :, -1].sort().indices.tolist()
                X = inputs[0,inds][:,:2]
                m = int(len(X)/2)
                plt.scatter(X[m:, 0], X[m:, 1], c='r', alpha=meta_pred.msk[0,inds][m:]/1.5+0.33)
                plt.scatter(X[:m, 0], X[:m, 1], c='b', alpha=meta_pred.msk[0,inds][:m]/1.5+0.33)
                px = [-20,20]
                py = [-(-20 * meta_output[0] + meta_output[2]) / meta_output[1],
                      -( 20 * meta_output[0] + meta_output[2]) / meta_output[1]]
                plt.plot(px, py)
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
                plt.show()

def plot_hist(hist, y_max, plot):
    """
    Plot the training behavior of the meta predictor.
    Args:
        hist (dict): A dictionary containing several information about the training
        y_max (float): A value for the top window of the y-axis (for messages and compression set)
        vis (int): Number of created predictors to visualize (plot) at the end of the training (only works for
                        linear classifier with 2-d datasets).
        plot (str): 'loss' or 'accuracy', the performance metric to plot
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(hist['epoch'], hist[f'train_{plot}'], c='b', lw=2)
    ax1.plot(hist['epoch'], hist[f'valid_{plot}'], c='b', lw=2, ls='--')
    ax1.plot(hist['epoch'], hist['bound_value'], c='b', lw=2, ls=':')
    ax2.plot(hist['epoch'], hist['n_sigma'], c='r', lw=1)
    ax2.plot(hist['epoch'], hist['n_Z'], c='r', lw=1, ls='--')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, y_max * 1.02)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(f'{plot}')
    ax2.set_ylabel('cardinality')
    ax1.legend(['Train', 'Valid', 'Bound'], loc=2)
    ax2.legend(['n_$\sigma$', 'n_Z'], loc=3)
    plt.show()