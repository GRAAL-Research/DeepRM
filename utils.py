from PIL import Image
import numpy as np
import random
from copy import copy
from matplotlib import pyplot as plt
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def lin_loss(output, targets, reduction='mean'):
    """
    Computes the total linear loss.
    Args:
        output (torch.tensor of size batch_size x m): The output (0 or 1) of the predictor
        targets (torch.tensor of size batch_size x m): The labels (0 or 1)
    Return:
        Float, the total linear loss incurred.
    """
    if reduction == 'mean':
        return torch.mean(((output * targets) + 1) / 2)
    if reduction is None:
        return ((output * targets) + 1) / 2
    assert False, 'Incorrect reduction parameter'


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

class Predictor():
    def __init__(self, d, pred, batch_size):
        super(Predictor, self).__init__()
        self.pred_type = pred[0]
        self.d, self.batch_size = d, batch_size
        self.num_param_init(d, pred)
        self.structure_init(self.batch_size)

    def num_param_init(self, d, pred):
        self.num_param = 0
        self.arch = 0
        if self.pred_type == 'linear_classif':
            self.num_param = d + 1
        if self.pred_type == 'small_nn':
            self.arch = [self.d] + pred[1] + [1]
            for i in range(1, len(self.arch)):
                self.num_param += (self.arch[i - 1] + 1) * self.arch[i]
    def structure_init(self, batch_size):
        self.structure = []
        if self.pred_type == 'small_nn':
            for k in range(batch_size):
                structure = torch.nn.ModuleList()
                for i in range(len(self.arch) - 1):
                    structure.append(nn.Linear(self.arch[i], self.arch[i + 1]))
                    if i < len(self.arch) - 2:
                        structure.append(nn.ReLU())
                self.structure.append(structure)

    def update_weights(self, weights):
        self.weights = weights
        if self.pred_type == 'small_nn':
            for i in range(len(weights)):
                count_1, count_2, j = 0, 0, 0
                for layer in self.structure[i]:
                    if isinstance(layer, nn.Linear):
                        count_2 += self.arch[j] * self.arch[j + 1]
                        layer.weight.data = torch.reshape(self.weights[i, count_1:count_2], (self.arch[j + 1], self.arch[j]))
                        count_1 += self.arch[j] * self.arch[j + 1]
                        count_2 += self.arch[j + 1]
                        layer.bias.data = torch.reshape(self.weights[i, count_1:count_2], (self.arch[j + 1],))
                        count_1 += self.arch[j + 1]
                        j += 1

    def forward(self, inputs, weights, return_sign = False):
        if self.pred_type == 'linear_classif':
            out = torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1) * self.weights[:, :-1], dim=-1) + self.weights[:, -1]
            out = torch.transpose(out, 0, 1)
        elif self.pred_type == 'small_nn':
            input = inputs[0, :, :-1]
            count_1, count_2, j = 0, 0, 0
            for layer in self.structure[0]:
                if isinstance(layer, nn.Linear):
                    count_2 += self.arch[j] * self.arch[j + 1]
                    W = torch.reshape(weights[0, count_1:count_2], (self.arch[j + 1], self.arch[j]))
                    count_1 += self.arch[j] * self.arch[j + 1]
                    count_2 += self.arch[j + 1]
                    b = torch.reshape(weights[0, count_1:count_2], (self.arch[j + 1],))
                    count_1 += self.arch[j + 1]
                    j += 1
                    input = torch.matmul(input, W.T) + b
                else:
                    input = layer(input)
            out = input
            for i in range(1,len(inputs)):
                input = inputs[i,:,:-1]
                count_1, count_2, j = 0, 0, 0
                for layer in self.structure[i]:
                    if isinstance(layer, nn.Linear):
                        count_2 += self.arch[j] * self.arch[j + 1]
                        W = torch.reshape(weights[i, count_1:count_2], (self.arch[j + 1], self.arch[j]))
                        count_1 += self.arch[j] * self.arch[j + 1]
                        count_2 += self.arch[j + 1]
                        b = torch.reshape(weights[i, count_1:count_2], (self.arch[j + 1],))
                        count_1 += self.arch[j + 1]
                        j += 1
                        input = torch.matmul(input, W.T) + b
                    else:
                        input = layer(input)
                out = torch.hstack((out, input))
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
    hist['train_acc'].append(values[0].cpu())
    hist['train_loss'].append(values[1].cpu())
    hist['valid_acc'].append(values[2].cpu())
    hist['valid_loss'].append(values[3].cpu())
    hist['test_acc'].append(values[4].cpu())
    hist['test_loss'].append(values[5].cpu())
    hist['bound_value'].append(values[6])
    hist['n_sigma'].append(values[7])
    hist['n_Z'].append(values[8])
    hist['epoch'].append(values[9])

def update_wandb(wandb, hist):
    wandb.log({'train_acc': hist['train_acc'][-1],
               'train_loss': hist['train_loss'][-1],
               'valid_acc': hist['valid_acc'][-1],
               'valid_loss': hist['valid_loss'][-1],
               'test_acc': hist['test_acc'][-1],
               'test_loss': hist['test_loss'][-1],
               'bound_value': hist['bound_value'][-1],
               'n_sigma': hist['n_sigma'][-1],
               'n_Z': hist['n_Z'][-1],
               'epoch': hist['epoch'][-1]})
def write(file_name, dict, hist, best_epoch):
    """
    Writes in a .txt file the hyperparameters and results of a training of the BGN algorithm
        on a given dataset.

    Args:
        file_name (str): The name of the .txt file to write into.
        algo (str): Name of the used algorithm.
        dataset (str): Name of the dataset.

    """
    keys = []
    for key in dict:
        keys.append(key)
    keys.sort()
    file = open("results/" + str(file_name) + ".txt", "a")
    for key in keys:
        file.write(str(dict[key])+"\t")
    file.write(str(hist['train_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['valid_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['test_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['bound_value'][best_epoch].item()) + "\t")
    file.write(str(hist['n_sigma'][best_epoch]) + "\t")
    file.write(str(hist['n_Z'][best_epoch]))
    file.write("\n")
    file.close()

def is_job_already_done(experiment_name, dict):
    cnt_nw = 0
    new, keys = [], []
    for key in dict:
        keys.append(key)
    keys.sort()
    for key in keys:
        new.append(str(dict[key]))
    try:
        with open("results/" + str(experiment_name) + ".txt", "r") as tes:
            tess = [line.strip().split('\t') for line in tes]
        tes.close()
        for i in range(len(tess)):
            if tess[i][:-6] == new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open("results/" + str(experiment_name) + ".txt", "a")
        for key in keys:
            file.write(key + "\t")
        file.write('train_acc' + "\t")
        file.write('valid_acc' + "\t")
        file.write('test_acc' + "\t")
        file.write('bound_value' + "\t")
        file.write('n_sigma' + "\t")
        file.write('n_Z' + "\n")
        file.close()
    return cnt_nw > 0

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

def show_decision_boundaries(meta_pred, dataset, data_loader, vis, pred, wandb, DEVICE):
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
        examples = []
        for inputs, targets in data_loader:
            for j in range(len(inputs)):
                if i < vis:
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()
                    i += 1
                    inputs, targets = inputs.float(), targets.float()
                    inds = inputs[j, :, -1].sort().indices.tolist()
                    X = inputs[j, inds][:,:2]
                    m = int(len(X)/2)
                    plt.scatter(X[m:, 0].cpu(), X[m:, 1].cpu(), c='r', alpha=meta_pred.msk[j,inds][m:].cpu()/1.5+0.33)
                    plt.scatter(X[:m, 0].cpu(), X[:m, 1].cpu(), c='b', alpha=meta_pred.msk[j,inds][:m].cpu()/1.5+0.33)
                    if str(DEVICE) == 'gpu':
                        inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                    meta_output = meta_pred(inputs)[j]
                    if pred.pred_type == 'linear_classif':
                        px = [-20,20]
                        py = [-(-20 * meta_output[0] + meta_output[2]) / meta_output[1],
                              -( 20 * meta_output[0] + meta_output[2]) / meta_output[1]]
                        plt.plot(px, py)
                    if pred.pred_type == 'small_nn':
                        h = .05  # step size in the mesh
                        x_min, x_max = X[:, 0].cpu().min() - 10, X[:, 0].cpu().max() + 10
                        y_min, y_max = X[:, 1].cpu().min() - 10, X[:, 1].cpu().max() + 10
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                             np.arange(y_min, y_max, h))
                        input = np.array(np.c_[xx.ravel(), yy.ravel()])
                        input = np.hstack((input,np.ones((len(input),1))))
                        input = torch.from_numpy(np.array([input]))
                        if str(DEVICE) == 'gpu':
                            input = input.cuda()
                        Z = pred.forward(input.double(), torch.reshape(meta_output, (1,-1)).double())
                        Z = torch.round(Z.reshape(xx.shape)).cpu()
                        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
                    if dataset in ['easy', 'hard']:
                        plt.xlim(-20, 20)
                        plt.ylim(-20, 20)
                    if dataset == 'moons':
                        plt.xlim(torch.mean(X[:, 0].cpu())-10, torch.mean(X[:, 0].cpu())+10)
                        plt.ylim(torch.mean(X[:, 1].cpu())-10, torch.mean(X[:, 1].cpu())+10)
                    plt.savefig(f"figures/decision_boundaries/decision_boundaries_{i}.png")
                    if wandb is not None:
                        im_frame = Image.open(f"figures/decision_boundaries/decision_boundaries_{i}.png")
                        image = wandb.Image(np.array(im_frame), caption=f"decision_boundaries/decision_boundaries_{i}") # file_type="jpg"
                        examples.append(image)
    if wandb is not None:
        wandb.log({"Decision boundaries": examples})
    #plt.show()

def show_loss_acc(loss, acc, epoch, wandb):
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
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    plt.scatter(acc[0].cpu()-0.1+(torch.rand(acc[0].shape)-0.5)/20, torch.clamp(loss[0].cpu(), 0, 100), c='b', s=2)
    plt.scatter(acc[1].cpu()+(torch.rand(acc[1].shape)-0.5)/20, torch.clamp(loss[1].cpu(), 0, 100), c='y', s=2)
    plt.scatter(acc[2].cpu()+0.1+(torch.rand(acc[2].shape)-0.5)/20, torch.clamp(loss[2].cpu(), 0, 100), c='g', s=2)
    plt.legend(['Train', 'Valid', 'Test'])
    plt.xlabel('Accuracy')
    plt.ylabel('BCE-Loss')
    plt.ylim(0,100)
    plt.savefig(f"figures/loss_acc/loss_acc_{epoch}.png")
    if wandb is not None:
        im_frame = Image.open(f"figures/loss_acc/loss_acc_{epoch}.png")
        image = wandb.Image(np.array(im_frame), caption=f"loss_acc/loss_acc_{epoch}") # file_type="jpg"
        wandb.log({f"Loss / acc. ({epoch})": image})

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
    ax1.plot(hist['epoch'], hist[f'test_{plot}'], c='b', lw=2, ls='--')
    ax1.plot(hist['epoch'], hist['bound_value'], c='b', lw=2, ls=':')
    ax2.plot(hist['epoch'], hist['n_sigma'], c='r', lw=1)
    ax2.plot(hist['epoch'], hist['n_Z'], c='r', lw=1, ls='--')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, y_max * 1.02)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(f'{plot}')
    ax2.set_ylabel('cardinality')
    ax1.legend(['Train', 'Test', 'Bound'], loc=2)
    ax2.legend(['n_$\sigma$', 'n_Z'], loc=3)
    plt.savefig("figures/hist.png")