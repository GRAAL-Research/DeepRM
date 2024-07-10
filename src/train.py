from copy import copy
from time import time

import torch
import torch.nn as nn
import wandb
from loguru import logger

from src.data.loaders import train_valid_loaders
from src.model.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.result.compute_stats import stats
from src.result.decision_boundaries import show_decision_boundaries
from src.result.history import update_hist, update_wandb
from src.utils import create_run_name


def train(meta_pred: SimpleMetaNet, pred: Predictor, data, optimizer, scheduler, criterion, pen_msg, task_dict: dict,
          is_sending_wandb_last_run_alert: bool):
    """
    Trains a meta predictor using PyTorch.

    Args:
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
            bound_computation (bool): whether to compute the bounds.
    Returns:
        tuple of: information about the model at the best training epoch (dictionary), best training epoch (int).
    """
    # Retrieving information
    balanced = task_dict['balanced']
    splits = task_dict['splits']
    seed = task_dict['seed']
    early_stopping_tolerance = task_dict['early_stopping_tolerance']
    early_stopping_patience = task_dict['early_stopping_patience']
    n_epoch = task_dict['n_epoch']
    batch_size = task_dict['batch_size']
    msg_size = task_dict['msg_size']
    msg_type = task_dict['msg_type']
    pen_msg_coef = task_dict['pen_msg_coef']
    device = task_dict['device']
    loss_power = task_dict['loss_power']
    bound_computation = task_dict['bound_computation']
    task = task_dict['task']
    valid_metric = "valid_acc" if task_dict['task'] == "classification" else "valid_loss"
    n_instances_per_class_per_dataset = task_dict["n_instances_per_dataset"] // 2

    torch.autograd.set_detect_anomaly(True)
    train_loader, valid_loader, test_loader, tr_var, vd_var, te_var = train_valid_loaders(data, batch_size, splits, seed=seed)
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
                inputs = inputs.float()[:, n_instances_per_class_per_dataset:]
                targets = (inputs.clone()[:, :, -1] + 1) / 2
                inputs, targets = inputs.float(), targets.float()
                if str(device) == 'gpu':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                optimizer.zero_grad()  # Zeroing the gradient everywhere in the meta-learner
                meta_output = meta_pred(inputs)  # Computing the parameters of the predictor.
                pred.set_weights(meta_output, len(inputs))  # Updating the weights of the predictor
                output, _ = pred.forward(inputs)  # Computing the predictions for the task
                if balanced or task == "regression":
                    loss = torch.mean(torch.mean(criterion(output, targets), dim=1) ** loss_power)
                else:
                    loss = 0
                    for batch in range(len(output)):
                        loss += (torch.mean(criterion(output[batch, targets[batch] == 0],
                                                      targets[batch, targets[batch] == 0])) / 2 +
                                 torch.mean(criterion(output[batch, targets[batch] == 1],
                                                      targets[batch, targets[batch] == 1])) / 2) ** loss_power
                    loss /= len(output)
                loss += pen_msg(meta_pred.msg, pen_msg_coef)  # Regularized loss
                loss.backward()  # Gradient computation
                optimizer.step()  # Backprop step
        # Computation of statistics about the current training epoch
        tr_acc, tr_loss, _ = stats(task, meta_pred, pred, criterion, train_loader, msg_type, bound_computation, device)
        vd_acc, vd_loss, _ = stats(task, meta_pred, pred, criterion, valid_loader, msg_type, bound_computation, device)
        te_acc, te_loss, bound = stats(task, meta_pred, pred, criterion, test_loader, msg_type,bound_computation,device)
        update_hist(hist, (tr_acc, tr_loss, vd_acc, vd_loss, te_acc, te_loss, bound, msg_size,
                           meta_pred.comp_set_size, i))  # Tracking results
        rolling_val_perf = torch.mean(torch.tensor(hist[valid_metric][-min(100, i + 1):]))
        if task_dict["is_using_wandb"]:
            update_wandb(wandb, hist)  # Upload information to WandB
        epo = '0' * (i + 1 < 100) + '0' * (i + 1 < 10) + str(i + 1)
        if task == "classification":
            print(f'Epoch {epo} - Train acc: {tr_acc:.4f} - Val acc: {vd_acc:.4f} - Test acc: {te_acc:.4f} - '
                  f'Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), '
                  f'(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - begin)}')  # Print information in console
        elif task == "regression":
            print(f'Epoch {epo} - Train R2: {1 - tr_loss / tr_var:.4f} - Val R2: {1 - vd_loss / vd_var:.4f} - '
                  f'Test R2: {1 - te_loss / te_var:.4f} - '
                  f'Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), '
                  f'(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - begin)}')  # Print information in console
        scheduler.step(rolling_val_perf)  # Scheduler step
        if i == 1 or rolling_val_perf > best_rolling_val_acc + early_stopping_tolerance:  # If an improvement has been done in validation...
            best_epoch = copy(i)  # ...We keep track of it
            best_rolling_val_acc = copy(rolling_val_perf)
        if ((tr_acc < 0.525 and i > 50) or  # If no learning has been made...
                i - best_epoch > early_stopping_patience):  # ... or no improvements for a while ...
            logger.info("The early stopping stopped the training.")
            break  # Early stopping is made

    if task_dict['n_features'] == 2 and task_dict["is_using_wandb"]:
        show_decision_boundaries(meta_pred, task_dict['dataset'], test_loader, pred, wandb, device)

    if task_dict["is_using_wandb"]:
        if is_sending_wandb_last_run_alert and task_dict["is_wandb_alert_activated"]:
            wandb.alert(title="✅ Done", text="The experiment is over.")
        wandb.finish()

    return hist, best_epoch
