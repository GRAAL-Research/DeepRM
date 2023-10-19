# Copyright 2023 Benjamin Leblanc, Alexandre Drouin, Mathieu Bazinet, Pascal Germain, Valentina Zantedeschi
import matplotlib.pyplot as plt

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

def deeprm(dataset = 'easy_gauss_lin',
           seed = 0,
           n = 1000,
           m = 10,
           d = 2,
           train_split = 0.8,
           meta_predictor = 'simplenet',
           predictor = 'linear_classif',
           kern_1_dim = [1000, 500, 10],
           kern_2_dim = [1000, 500, 10],
           modl_1_dim = [1000, 1000, 500, 2], # Last value: message size
           modl_2_dim = [1000, 1000, 500, 10],
           k = 2,
           tau = 10,
           init = 'kaiming_unif',
           criterion = 'bce_loss',
           start_lr = 1e-4,
           pen_msg = 'l1',
           pen_msg_coef = 0,#1e2,
           patience = 5,
           factor = 0.5,
           tol = 1e-2,
           early_stop = 20,
           optimizer = 'adam',
           scheduler = 'plateau',
           n_epoch = 200,
           batch_size = 50,
           DEVICE = 'cpu',
           bound_type = 'Mathieu',
           vis = 5,
           plot = 'acc'
    ):
    set_seed(seed)
    if dataset == 'easy_gauss_lin':
        data = data_gen(n, m, d, 'easy', True, False)
    elif dataset == 'hard_gauss_lin':
        data = data_gen(n, m, d, 'hard')
    if meta_predictor == 'simplenet':
        meta_pred = SimpleMetaNet(d, (kern_1_dim, kern_2_dim, modl_1_dim, modl_2_dim), d+1, m, d, k, tau, batch_size, init)
    if predictor == 'linear_classif':
        pred = lin_clas
    if criterion == 'bce_loss':
        crit = nn.BCELoss()
    if pen_msg == 'l1':
        penalty_msg = l1
    if optimizer == 'adam':
        opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(start_lr))
    if scheduler == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti,
                                                           mode='max',
                                                           factor=factor,
                                                           patience=patience,
                                                           threshold=tol,
                                                           verbose=False)
    hist = train(meta_pred, pred, data, train_split, opti, sched, tol, early_stop,
                 n_epoch, batch_size, crit, penalty_msg, pen_msg_coef, vis, bound_type, DEVICE)
    if plot in ['loss', 'acc']:
        plot_hist(hist, modl_1_dim[-1]-m, plot)
deeprm()
