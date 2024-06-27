import math
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from copy import copy
from bound import compute_bound
from time import time
import wandb
from loguru import logger

from wandb_utils import create_run_name


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

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def train(meta_pred, pred, data, optimizer, scheduler, criterion, pen_msg, task_dict: dict):
    """
    Trains a meta predictor using PyTorch.

    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (Predictor): A predictor whose parameters are computed by the meta predictor.
        data (Dataset): A dataset.
        optimizer (torch.optim): meta-neural network optimizer;
        scheduler (torch.optim.lr_scheduler): learning rate decay scheduler;
        criterion (function): loss function (choices: 'bce_loss');
        pen_msg (function): message penalty function;
        task_dict (dictionary) containing the following:
            splits ([float, float, float]): train, valid and test proportion of the data;
            early_stopping_tolerance (float): Quantity by which the loss must diminish in order for this increment not to be marginal
            early_stopping_patience (int): Number of epochs by which, if the loss hasn't diminished by « early_stopping_tolerance », the train stops
            n_epoch (int): The maximum number of epochs.
            batch_size (int): Batch size.
            pen_msg_coef (float): Message regularization factor.
            device (str): whether to use the gpu (choices: 'gpu', 'cpu');
    Returns:
        tuple of: information about the model at the best training epoch (dictionary), best training epoch (int).
    """
    # Retrieving information
    splits = task_dict['splits']
    early_stopping_tolerance = task_dict['early_stopping_tolerance']
    early_stopping_patience = task_dict['early_stopping_patience']
    n_epoch = task_dict['n_epoch']
    batch_size = task_dict['batch_size']
    msg_size = task_dict['msg_size']
    msg_type = task_dict['msg_type']
    pen_msg_coef = task_dict['pen_msg_coef']
    device = task_dict['device']
    m = meta_pred.m

    torch.autograd.set_detect_anomaly(True)
    train_loader, valid_loader, test_loader = train_valid_loaders(data, batch_size, splits)
    best_rolling_val_acc, best_epoch = 0, 0
    # The following information will be recorded at each epoch
    hist = {'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'test_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'test_acc': [],
            'bound_lin': [],
            'bound_hyp': [],
            'bound_kl': [],
            'bound_mrch': []}

    if task_dict["is_using_wandb"]:
        run_name = create_run_name(task_dict)
        wandb.init(name=run_name, project=task_dict["project_name"], config=task_dict)

    begin = time()
    for i in range(n_epoch):
        meta_pred.train()  # We put the meta predictor in training mode
        with (torch.enable_grad()):
            for inputs in train_loader:  # Iterating over the various batches
                targets = (inputs.clone()[:, :, -1] + 1) / 2
                inputs, targets = inputs.float(), targets.float()
                if str(device) == 'gpu':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                optimizer.zero_grad()  # Zeroing the gradient everywhere in the meta-learner
                meta_output = meta_pred(inputs[:, m:])  # Computing the parameters of the predictor.
                pred.update_weights(meta_output, len(inputs))  # Updating the weights of the predictor
                output = pred.forward(inputs[:, m:])  # Computing the predictions for the task
                loss = torch.mean(torch.mean(criterion(output, targets[:, m:]), dim=1) ** 0.5)
                loss += pen_msg(meta_pred.msg, pen_msg_coef)  # Regularized loss
                loss.backward()  # Gradient computation
                optimizer.step()  # Backprop step
        # Computation of statistics about the current training epoch
        tr_acc, tr_loss, _ = stats(meta_pred, pred, criterion, train_loader, msg_type, device)
        vd_acc, vd_loss, _ = stats(meta_pred, pred, criterion, valid_loader, msg_type, device)
        te_acc, te_loss, bound = stats(meta_pred, pred, criterion, test_loader, msg_type, device)
        update_hist(hist, (tr_acc, tr_loss, vd_acc, vd_loss, te_acc, te_loss, bound, msg_size,
                           meta_pred.comp_set_size, i))  # Tracking results
        rolling_val_acc = torch.mean(torch.tensor(hist['valid_acc'][-min(100, i + 1):]))
        if task_dict["is_using_wandb"]:
            update_wandb(wandb, hist)  # Upload information to WandB
        epo = '0' * (i + 1 < 100) + '0' * (i + 1 < 10) + str(i + 1)
        print(f'Epoch {epo} - Train acc: {tr_acc:.2f} - Val acc: {vd_acc:.2f} - Test acc: {te_acc:.2f} - '
              f'Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), '
              f'(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - begin)}')  # Print information in console
        scheduler.step(rolling_val_acc)  # Scheduler step
        if i == 1 or rolling_val_acc > best_rolling_val_acc + early_stopping_tolerance:  # If an improvement has been done in validation...
            best_epoch = copy(i)  # ...We keep track of it
            best_rolling_val_acc = copy(rolling_val_acc)
        if ((tr_acc < 0.525 and i > 50) or  # If no learning has been made...
                i - best_epoch > early_stopping_patience):  # ... or no improvements for a while ...
            logger.info("The early stopping stopped the training.")
            break  # Early stopping is made

    if task_dict['d'] == 2 and task_dict["is_using_wandb"]:
        show_decision_boundaries(meta_pred, task_dict['dataset'], test_loader, pred, wandb, device)

    if task_dict["is_using_wandb"]:
        wandb.finish()

    return hist, best_epoch


def stats(meta_pred, pred, criterion, data_loader, msg_type, device):
    """
    Computes the overall accuracy, loss and bounds of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (Predictor): A predictor whose parameters are computed by the meta predictor.
        criterion (torch.nn, function): Loss function
        data_loader (DataLoader): A DataLoader to test on.
        msg_type (str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
        device (str): 'cuda', or 'cpu'; whether to use the gpu
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    bnd_lin, bnd_hyp, bnd_kl, bnd_mrch = [], [], [], []  # The various bounds to compute
    meta_pred.eval()  # We put the meta predictor in evaluation mode
    m = meta_pred.m
    with torch.no_grad():
        i, k = 0, 0  # Number of batches / examples we have been through
        tot_loss, tot_acc = [], []
        for inputs in data_loader:
            targets = (inputs.clone()[:, :, -1] + 1) / 2
            n = copy(len(targets) * len(targets[0]))
            i += n
            k += 1
            inputs, targets = inputs.float(), targets.float()
            if str(device) == 'gpu':
                inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
            meta_output = meta_pred(inputs[:, :m])  # Computing the parameters of the predictor
            pred.update_weights(meta_output, len(inputs))  # Updating the weights of the predictor
            output = pred.forward(inputs[:, m:], True)  # Computing the predictions for the task
            loss = criterion(output[0], targets[:, m:])  # Loss computation
            tot_loss.append(torch.sum(loss).cpu())
            acc = m * lin_loss(output[1], targets[:, m:] * 2 - 1)  # Accuracy computation
            tot_acc.append(torch.mean(acc / m).cpu())
            if msg_type is not None:
                for b in range(len(inputs)):  # For all datasets, we compute the bounds
                    bnds = compute_bound(['linear', 'hyperparam', 'kl', 'marchand'], meta_pred, pred, m,
                                         m - acc.item(), 0.05, 0, 1, inputs[[b], m:], targets[[b], m:])
                    bnd_lin.append(bnds[0])
                    bnd_hyp.append(bnds[1])
                    bnd_kl.append(bnds[2])
                    bnd_mrch.append(bnds[3])
        if msg_type is None:
            return np.mean(tot_acc), np.mean(tot_loss), []
        # We only return the mean bound obtained on the various datasets
        return np.mean(tot_acc), np.mean(tot_loss), [np.mean(bnd_lin), np.mean(bnd_hyp), np.mean(bnd_kl),
                                                     np.mean(bnd_mrch)]
