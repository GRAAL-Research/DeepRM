# Copyright 2024 Benjamin Leblanc, Alexandre Drouin, Mathieu Bazinet, Nathaniel D'Amours, Pascal Germain
# This file is part of the Deep Reconstruction Machine (DeepRM) work.

# DeepRM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from train import *
from datasets import *
from copy import copy
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid


def deeprm(experiment_name, dataset, seed, n, m, d, splits, meta_pred, pred_arch, comp_set_size, msg_size, ca_dim,
           kme_dim, mod_1_dim, mod_2_dim, tau, init, criterion, start_lr, pen_msg, pen_msg_coef, msg_type, batch_size,
           patience, factor, tol, early_stop, optimizer, scheduler, n_epoch, device, weightsbiases):
    """
    Args:
        experiment_name: (list of str) experiment name;
        dataset (list of str): datasets (choices: 'moons', 'easy', 'both');
        seed (list of int): random seeds;
        n (list of int): total number of datasets
        m (list of int): number of examples per dataset;
        d (list of int): dimension of each example;
        splits (list of [float, float, float]): train, valid and test proportion of the data;
        meta_pred (list of str): meta_predictor to use (choices: 'simplenet');
        pred_arch (list of list of int): architecture of the predictor (a ReLU network);
        comp_set_size (list of int): compression set size;
        msg_size (list of int): message size;
        ca_dim (list of list of int): custom attention's MLP architecture;
        kme_dim (list of list of int): KME's MLP architecture;
        mod_1_dim (list of list of int): MLP #1 architecture;
        mod_2_dim (list of list of int): MLP #2 architecture;
        tau (list of int): temperature parameter (softmax in custom attention);
        init (list of str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
        criterion (list of str): loss function (choices: 'bce_loss');
        start_lr (list of float): initial learning rate;
        pen_msg (list of str): type of message penalty (choices: 'l1', 'l2');
        pen_msg_coef (list of float): message penalty coefficient;
        msg_type (list of str): type of message (choices: 'dsc' (discret), 'cnt' (continuous));
        batch_size (list of int): batch size;
        scheduler (list of str): learning rate decay type (choices: 'plateau');
        patience (list of int): learning rate decay patience;
        factor (list of float): factor by which the learning rate is multiplied (one decay step);
        tol (list of float): tolerance for learning rate decay to apply;
        early_stop (list of int): early stopping number of epoch;
        optimizer (list of str): meta-neural network optimizer (choices: 'adam', 'rmsprop');
        n_epoch (list of int): maximum number of epoch for the training phase;
        device (list of str): device on which to compute (choices: 'cpu', 'gpu');
        weightsbiases (list of [str, str]): list with WandB team and project if data is to be stocked on WandB;
                      (empty list): if data is not to be stocked in WandB.
    """

    # Creating a parameter grid for iterating on the different hyperparameters combinations.
    param_grid = ParameterGrid([{'dataset': dataset,
                                 'seed': seed,
                                 'n': n,
                                 'm': m,
                                 'd': d,
                                 'splits': splits,
                                 'meta_pred': meta_pred,
                                 'pred_arch': pred_arch,
                                 'comp_set_size': comp_set_size,
                                 'msg_size': msg_size,
                                 'ca_dim': ca_dim,
                                 'kme_dim': kme_dim,
                                 'mod_1_dim': mod_1_dim,
                                 'mod_2_dim': mod_2_dim,
                                 'tau': tau,
                                 'init': init,
                                 'criterion': criterion,
                                 'start_lr': start_lr,
                                 'pen_msg': pen_msg,
                                 'pen_msg_coef': pen_msg_coef,
                                 'msg_type': msg_type,
                                 'batch_size': batch_size,
                                 'scheduler': scheduler,
                                 'patience': patience,
                                 'factor': factor,
                                 'tol': tol,
                                 'early_stop': early_stop,
                                 'optimizer': optimizer,
                                 'n_epoch': n_epoch,
                                 'device': device,
                                 'weightsbiases': weightsbiases}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset)}
    param_grid = sorted(param_grid, key=lambda p: ordering[p['dataset']])
    n_tasks = len(param_grid)
    meta_pred, data, opti, sched, crit, penalty_msg, task_dict = None, None, None, None, None, None, None

    for i, task_dict in enumerate(param_grid):  # Iterating on the different hyperparameters combinations.
        if task_dict['dataset'] == 'mnist':
            task_dict['n'], task_dict['m'], task_dict['d'] = 90, 6313*2, 784  # For non-synthetic data, these are fixed
        if task_dict['msg_size'] == 0:
            task_dict['msg_type'] = 'none'
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        if is_job_already_done(experiment_name, task_dict): # Verify if this hyp. param. comb. has already been tested
            print("Already done; passing...\n")
        elif task_dict['msg_type'] == 'dsc' and task_dict['pen_msg_coef'] > 0:  # Passes on incoherent hyp. param. comb.
            print("Doesn't make sens to regularize discrete messages; passing...\n")
        elif task_dict['comp_set_size'] + task_dict['msg_size'] == 0:   # Passes on incoherent hyp. param. comb.
            print("Opaque network; passing...\n")
        else:   # The current hyp. param. comb. will be tested
            set_seed(task_dict['seed']) # Sets the random seed for numpy, torch and random packages
            if task_dict['dataset'] in ['moons', 'easy', 'hard']:  # Generating the datasets
                data = data_gen(task_dict['dataset'], task_dict['n'], task_dict['m'], task_dict['d'], True)
            elif task_dict['dataset'] == 'mnist':
                pass    # TODO

            pred = Predictor(task_dict['d'], task_dict['pred_arch'], task_dict['batch_size'])   # Predictor init.
            if task_dict['meta_pred'] == 'simplenet':  # Meta-predictor initialization
                meta_pred = SimpleMetaNet(pred.num_param, task_dict)
            if task_dict['criterion'] == 'bce_loss':    # Criterion initialization
                crit = nn.BCELoss(reduction='none')
            if task_dict['pen_msg'] == 'l1':    # Message penalty initialization
                penalty_msg = l1
            elif task_dict['pen_msg'] == 'l2':
                penalty_msg = l2
            if task_dict['optimizer'] == 'adam':    # Optimizer initialization
                opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            elif task_dict['optimizer'] == 'rmsprop':
                opti = torch.optim.RMSprop(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            if task_dict['scheduler'] == 'plateau':  # Scheduler initialization
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti,
                                                                   mode='max',
                                                                   factor=task_dict['factor'],
                                                                   patience=task_dict['patience'],
                                                                   threshold=task_dict['tol'],
                                                                   verbose=True)
            hist, best_epoch = train(meta_pred, pred, data, opti, sched, crit, penalty_msg, task_dict)  # Launch train.
            write(experiment_name, task_dict, hist, best_epoch)  # Training details are written in a .txt file


# Experiments launcher
deeprm(experiment_name='Test_wandb_4',
       dataset=['moons'],
       seed=[0],
       n=[4000],
       m=[100],
       d=[2],
       splits=[[0.55, 0.20, 0.25]],
       meta_pred=['simplenet'],
       pred_arch=[[3],
                  [7]],
       comp_set_size=[0,8,15],
       msg_size=[0,5,10],
       ca_dim=[[100],
               [100, 100]],
       kme_dim=[[100],
                [100, 100]],
       mod_1_dim=[[100]],
       mod_2_dim=[[100],
                  [100, 50]],
       tau=[20],
       init=['kaiming_unif'],
       criterion=['bce_loss'],
       start_lr=[1e-2, 1e-3],
       pen_msg=['l2'],
       pen_msg_coef=[0],
       msg_type=['dsc', 'cnt'],
       batch_size=[100],
       scheduler=['plateau'],
       patience=[100],
       factor=[0.5],
       tol=[1e-2],
       early_stop=[100],
       optimizer=['adam'],
       n_epoch=[2],
       device=['cpu'],
       weightsbiases=[[]]#['graal_deeprm2024', 'deeprm_attention_5']]
       )
