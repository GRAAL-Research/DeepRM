from copy import copy

import numpy as np
import torch

from src.bound.compute_bound import compute_bound
from src.model.predictor import Predictor
from src.model.utils.loss import lin_loss


def stats(meta_pred, pred: Predictor, criterion, data_loader, msg_type, device):
    """
    Computes the overall accuracy, loss and bounds of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
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
            pred.set_weights(meta_output, len(inputs))  # Updating the weights of the predictor
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
