from copy import copy

import numpy as np
import torch

from src.bound.compute_bound import compute_bound
from src.model.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.model.utils.loss import lin_loss


def stats(task, meta_pred: SimpleMetaNet, pred: Predictor, criterion, data_loader, msg_type, is_bound_computed, device):
    """
    Computes the overall accuracy, loss and bounds of a predictor on given task and dataset.
    Args:
        data_loader (DataLoader): A DataLoader to test on.
        msg_type (str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    bnd_lin, bnd_hyp, bnd_kl, bnd_mrch = [], [], [], []  # The various bounds to compute
    meta_pred.eval()  # We put the meta predictor in evaluation mode
    n_instances_per_class_per_dataset = meta_pred.n_instances_per_class_per_dataset
    with torch.no_grad():
        i, k = 0, 0  # Number of batches / examples we have been through
        tot_loss, tot_acc = [], []
        for inputs in data_loader:
            targets = (inputs.clone()[:, :, -1] + 1) / 2
            n = copy(len(targets) * len(targets[0]) / 2)
            i += n
            k += 1
            inputs, targets = inputs.float(), targets.float()
            if str(device) == 'gpu':
                inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
            meta_output = meta_pred(
                inputs[:, :n_instances_per_class_per_dataset])  # Computing the parameters of the predictor
            pred.set_weights(meta_output, len(inputs))  # Updating the weights of the predictor
            output = pred.forward(inputs[:, n_instances_per_class_per_dataset:])  # Computing the predictions for the task
            loss = criterion(output[0], targets[:, n_instances_per_class_per_dataset:])  # Loss computation
            tot_loss.append(torch.sum(loss).cpu())
            if task == "classification":
                acc = n_instances_per_class_per_dataset * lin_loss(output[1],
                                                                   targets[:, n_instances_per_class_per_dataset:] * 2 - 1)
                tot_acc.append(torch.mean(acc / n_instances_per_class_per_dataset).cpu())
                if is_bound_computed:
                    for b in range(len(inputs)):  # For all datasets, we compute the bounds
                        bnds = compute_bound(['linear', 'hyperparam', 'kl', 'marchand'], meta_pred, pred,
                                             n_instances_per_class_per_dataset,
                                             n_instances_per_class_per_dataset - acc.item(), 0.05, 0, 1,
                                             inputs[[b], n_instances_per_class_per_dataset:],
                                             targets[[b], n_instances_per_class_per_dataset:])
                        bnd_lin.append(bnds[0])
                        bnd_hyp.append(bnds[1])
                        bnd_kl.append(bnds[2])
                        bnd_mrch.append(bnds[3])
        if task == "classification":
            if is_bound_computed:
                # We only return the mean bound obtained on the various datasets
                return np.mean(tot_acc), np.sum(tot_loss) / i, \
                       [np.mean(bnd_lin), np.mean(bnd_hyp), np.mean(bnd_kl), np.mean(bnd_mrch)]
            else:
                return np.mean(tot_acc), np.sum(tot_loss) / i, [np.array(0.0), np.array(0.0), np.array(0.0), np.array(0.0)]
        if task == "regression":
            if is_bound_computed:
                # We only return the mean bound obtained on the various datasets
                return np.array(0.0), np.sum(tot_loss) / i, \
                       [np.mean(bnd_lin), np.mean(bnd_hyp), np.mean(bnd_kl), np.mean(bnd_mrch)]
            else:
                return np.array(0.0), np.sum(tot_loss) / i, [np.array(0.0), np.array(0.0), np.array(0.0), np.array(0.0)]
