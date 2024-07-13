from copy import copy
from time import time
from typing import Callable

import numpy as np
import torch
import wandb
from loguru import logger
from torch import nn

from src.data.loaders import train_valid_loaders
from src.model.predictor import Predictor
from src.result.compute_stats import stats
from src.result.decision_boundaries import show_decision_boundaries
from src.result.history import update_hist, update_wandb
from src.utils import create_run_name


def train_meta_predictor(meta_pred: nn.Module, pred: Predictor, datasets: np.ndarray, optimizer: torch.optim.Optimizer,
                         scheduler: torch.optim.lr_scheduler, criterion: nn.Module,
                         message_penalty_function: Callable[[torch.tensor, float], torch.tensor], config: dict,
                         is_sending_wandb_last_run_alert: bool):
    """
        config (dictionary) containing the following:
            batch_size (int): Batch size.
            bound_computation (bool): whether to compute the bounds.
    Returns:
        tuple of: information about the model at the best training epoch (dictionary), best training epoch (int).
    """
    # Retrieving information
    is_dataset_balanced = config['is_dataset_balanced']
    splits = config['splits']
    seed = config['seed']
    early_stopping_tolerance = config['early_stopping_tolerance']
    early_stopping_patience = config['early_stopping_patience']
    batch_size = config['batch_size']
    msg_size = config['msg_size']
    msg_type = config['msg_type']
    msg_penalty_coef = config['msg_penalty_coef']
    device = config['device']
    loss_exponent = config['loss_exponent']
    bound_computation = config['bound_computation']
    task = config['task']
    valid_metric = "valid_acc" if config['task'] == "classification" else "valid_loss"
    n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2

    torch.autograd.set_detect_anomaly(True)
    train_loader, valid_loader, test_loader, tr_var, vd_var, te_var = train_valid_loaders(datasets, batch_size, splits,
                                                                                          seed=seed)
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

    if config["is_using_wandb"]:
        run_name = create_run_name(config)
        wandb.init(name=run_name, project=config["project_name"], config=config)

    begin = time()
    for i in range(config["max_epoch"]):
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
                if is_dataset_balanced or task == "regression":
                    loss = torch.mean(torch.mean(criterion(output, targets), dim=1) ** loss_exponent)
                else:
                    loss = 0
                    for batch in range(len(output)):
                        loss += (torch.mean(criterion(output[batch, targets[batch] == 0],
                                                      targets[batch, targets[batch] == 0])) / 2 +
                                 torch.mean(criterion(output[batch, targets[batch] == 1],
                                                      targets[batch, targets[batch] == 1])) / 2) ** loss_exponent
                    loss /= len(output)
                loss += message_penalty_function(meta_pred.msg, msg_penalty_coef)  # Regularized loss
                loss.backward()  # Gradient computation
                optimizer.step()  # Backprop step
        # Computation of statistics about the current training epoch
        tr_acc, tr_loss, _ = stats(task, meta_pred, pred, criterion, train_loader, msg_type, bound_computation, device)
        vd_acc, vd_loss, _ = stats(task, meta_pred, pred, criterion, valid_loader, msg_type, bound_computation, device)
        te_acc, te_loss, bound = stats(task, meta_pred, pred, criterion, test_loader, msg_type, bound_computation,
                                       device)
        update_hist(hist, (tr_acc, tr_loss, vd_acc, vd_loss, te_acc, te_loss, bound, msg_size,
                           meta_pred.compression_set_size, i))  # Tracking results
        rolling_val_perf = torch.mean(torch.tensor(hist[valid_metric][-min(100, i + 1):]))
        if config["is_using_wandb"]:
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

    if config['n_features'] == 2 and config["is_using_wandb"]:
        show_decision_boundaries(meta_pred, config['dataset'], test_loader, pred, wandb, device)

    if config["is_using_wandb"]:
        if is_sending_wandb_last_run_alert and config["is_wandb_alert_activated"]:
            wandb.alert(title="✅ Done", text="The experiment is over.")
        wandb.finish()

    return hist, best_epoch
