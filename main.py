# Copyright 2023 Benjamin Leblanc, Alexandre Drouin, Mathieu Bazinet, Pascal Germain
# This file is part of the Deep Reconstruction Machine (DeepRM) work.

# DeepRM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from train import *
from datasets import *
from utils import plot_hist
from copy import copy
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
import wandb

def deeprm(experiment_name = 'Test_wandb_3',
           dataset = ['moons'], # easy, hard, moons, mnist
           seed = [7,8,9],
           n = [4000],  # Total number of datasets
           m = [60],   # Number of example per dataset
           d = [2],     # Dimension of each example
           splits = [[0.55, 0.20, 0.25]], # Train, valid and test proportion of the data
           train_splits = [[0.9, 10]], # Proportion of meta_learner food VS predictor food; number of examples chosen
           meta_predictr = ['simplenet'],                   # per batch for the meta-learner to learn on
           predictr = [['small_nn', [3]],
                       ['small_nn', [7]]
                       #['linear_classif', []],
                       ],
           kme_1_dim = [[200, 20]],
           kme_2_dim = [[200, 30]],
           modl_1_dim = [[100, 40]],
           modl_2_dim = [[100, 40]],
           modl_3_dim = [[200, 200, 0],
                         [200, 200, 2],
                         [200, 200, 5],
                         [200, 200, 10]], # Last value: message size
           modl_4_dim = [[200, 200, 30]],
           k = [0,2,4,6,8], # Either an integer (exact number of compression set), or float (threshold in Gumbel)
           tau = [1], # Temperature parameter (Gumbel)
           init = ['kaiming_unif'],
           criterion = ['bce_loss'],
           start_lr = [1e-3],
           pen_msg = ['l1'],
           pen_msg_coef = [0],#1e2,
           batch_size = [200],
           patience = [100],
           factor = [0.5],
           tol = [1e-2],
           early_stop = [100],
           optimizer = ['adam'],
           scheduler = ['plateau'],
           n_epoch = [1600],
           DEVICE = ['cpu'], # 'gpu' or 'cpu'
           bound_type = ['Alex'],
           independent_food = [False],
           vis = 16,
           vis_loss_acc = True,
           plot = None,
           weightsbiases = ['graal_deeprm2024', 'deeprm2'] # []
    ):
    weightsbiases.append(1)
    param_grid = ParameterGrid([{'dataset': dataset,
                                   'seed': seed,
                                   'n': n,
                                   'm': m,
                                   'd': d,
                                   'splits': splits,
                                   'train_splits': train_splits,
                                   'meta_predictor': meta_predictr,
                                   'predictor': predictr,
                                   'kme_1_dim': kme_1_dim,
                                   'kme_2_dim': kme_2_dim,
                                   'modl_1_dim': modl_1_dim,
                                   'modl_2_dim': modl_2_dim,
                                   'modl_3_dim': modl_3_dim,
                                   'modl_4_dim': modl_4_dim,
                                   'k': k,
                                   'tau': tau,
                                   'init': init,
                                   'criterion': criterion,
                                   'start_lr': start_lr,
                                   'pen_msg': pen_msg,
                                   'pen_msg_coef': pen_msg_coef,
                                   'batch_size': batch_size,
                                    'patience': patience,
                                    'factor': factor,
                                    'tol': tol,
                                    'early_stop': early_stop,
                                    'optimizer': optimizer,
                                    'scheduler': scheduler,
                                    'n_epoch': n_epoch,
                                    'DEVICE': DEVICE,
                                    'bound_type': bound_type,
                                    'independent_food': independent_food}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset']])
    n_tasks = len(param_grid)
    for i, task_dict in enumerate(param_grid):
        if task_dict['dataset'] in ['mnist']:
            task_dict['n'], task_dict['m'], task_dict['d'] = 90, 6313 * 2, 784
        if not task_dict['independent_food']:
            task_dict['train_splits'] = []
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        if is_job_already_done(experiment_name, task_dict):
            print("Already done; passing...\n")
        elif task_dict['k'] == task_dict['modl_3_dim'][-1] == 0:
            print("DeepRM is opac!...\n")
        else:
            set_seed(task_dict['seed'])
            if task_dict['dataset'] in ['moons', 'easy', 'hard']:
                data = data_gen(task_dict['n'], task_dict['m'], task_dict['d'], task_dict['dataset'], True, min(0,copy(vis)))
            elif task_dict['dataset'] in ['mnist']:
                data = load_mnist()
            pred = Predictor(task_dict['d'], task_dict['predictor'], task_dict['batch_size'])
            output_dim = pred.num_param
            set_seed(task_dict['seed'])
            if task_dict['meta_predictor'] == 'simplenet':
                if task_dict['independent_food']:
                    di = task_dict['train_splits'][-1]
                else:
                    di = task_dict['m']
                meta_pred = SimpleMetaNet(task_dict['d'],
                                            (task_dict['kme_1_dim'],
                                                  task_dict['kme_2_dim'],
                                                  task_dict['modl_1_dim'].copy(),
                                                  task_dict['modl_2_dim'].copy(),
                                                  task_dict['modl_3_dim'].copy(),
                                                  task_dict['modl_4_dim'].copy()),
                                          output_dim,
                                          di,
                                          task_dict['d'],
                                          task_dict['k'],
                                          task_dict['tau'],
                                          task_dict['batch_size'],
                                          task_dict['init'],
                                          task_dict['DEVICE'])

            if task_dict['criterion'] == 'bce_loss':
                crit = nn.BCELoss(reduction='none')
            if task_dict['pen_msg'] == 'l1':
                penalty_msg = l1
            if task_dict['optimizer'] == 'adam':
                opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            elif task_dict['optimizer'] == 'rmsprop':
                opti = torch.optim.RMSprop(meta_pred.parameters(), lr=copy(task_dict['start_lr']))
            if task_dict['scheduler'] == 'plateau':
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti,
                                                                   mode='max',
                                                                   factor=task_dict['factor'],
                                                                   patience=task_dict['patience'],
                                                                   threshold=task_dict['tol'],
                                                                   verbose=False)
            wandb_dico = {
                            'dataset': task_dict['dataset'],
                            'seed': task_dict['seed'],
                            'n': task_dict['n'],
                            'm': task_dict['m'],
                            'd': task_dict['d'],
                            'splits': task_dict['splits'],
                            'train_splits': task_dict['train_splits'],
                            'meta_predictor': task_dict['meta_predictor'],
                            'predictor': task_dict['predictor'],
                            'kme_1_dim': task_dict['kme_1_dim'],
                            'kme_2_dim': task_dict['kme_2_dim'],
                            'modl_1_dim': task_dict['modl_1_dim'],
                            'modl_2_dim': task_dict['modl_2_dim'],
                            'modl_3_dim': task_dict['modl_3_dim'],
                            'modl_4_dim': task_dict['modl_4_dim'],
                            'k': task_dict['k'],
                            'tau': task_dict['tau'],
                            'init': task_dict['init'],
                            'criterion': task_dict['criterion'],
                            'start_lr': task_dict['start_lr'],
                            'pen_msg': task_dict['pen_msg'],
                            'pen_msg_coef': task_dict['pen_msg_coef'],
                            'batch_size': task_dict['batch_size'],
                            'patience': task_dict['patience'],
                            'factor': task_dict['factor'],
                            'tol': task_dict['tol'],
                            'early_stop': task_dict['early_stop'],
                            'optimizer': task_dict['optimizer'],
                            'scheduler': task_dict['scheduler'],
                            'n_epoch': task_dict['n_epoch'],
                            'DEVICE': task_dict['DEVICE'],
                            'bound_type': task_dict['bound_type'],
                            'independent_food': task_dict['independent_food']
                        }
            weightsbiases[-1] = wandb_dico
            hist, best_epoch = train(meta_pred, pred, data, task_dict['dataset'], task_dict['splits'], task_dict['train_splits'], opti, sched, task_dict['tol'], task_dict['early_stop'], task_dict['n_epoch'],
                         task_dict['batch_size'], crit, penalty_msg, task_dict['pen_msg_coef'], vis, vis_loss_acc, task_dict['bound_type'], task_dict['DEVICE'], task_dict['independent_food'], weightsbiases)
            write(experiment_name, task_dict, hist, best_epoch-1)
            if plot in ['loss', 'acc']:
                plot_hist(hist, task_dict['modl_1_dim'][-1]-task_dict['m'], plot)
deeprm()