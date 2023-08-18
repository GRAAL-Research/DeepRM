# Copyright 2023 Benjamin Leblanc, Mathieu Bazinet, Alexandre Drouin, Pascal Germain, Valentina Zantedeschi
import matplotlib.pyplot as plt

# This file is part of the work Deep Reconstruction Machine (DeepRM).

# DeepRM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from train import *
from datasets import *
import torch

def deeprm(dataset = 'easy_gauss_lin',
           seed = 0,
           n = 1000,
           m = 20,
           d = 10,
           train_valid = [0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.05],
           meta_predictor = 'simplenet',
           predictor = 'linear_classif',
           kernel_dim = [200, 200, 10],
           hidden_dim = [200, 200],
           criterion = 'bce_loss',
           start_lr=1e-4,
           patience = 5,
           factor = 0.5,
           tol = 1e-2,
           early_stop = 20,
           optimizer = 'adam',
           scheduler = 'plateau',
           n_epoch = 200,
           batch_size = 50,
           DEVICE = "cpu",
           plot = 'loss'
    ):
    np.random.seed(seed)
    for train_split in train_valid:
        if dataset == 'easy_gauss_lin':
            data = data_gen(n, m, d, 'easy')
        elif dataset == 'hard_gauss_lin':
            data = data_gen(n, m, d, 'hard')
        if meta_predictor == 'simplenet':
            meta_pred = SimpleNet(d, kernel_dim, hidden_dim, d+1, m, d, batch_size)
        if predictor == 'linear_classif':
            pred = lin_clas
        if criterion == 'bce_loss':
            crit = nn.BCELoss(reduction='sum')
        if optimizer == 'adam':
            opti = torch.optim.Adam(meta_pred.parameters(), lr=copy.copy(start_lr))
        if scheduler == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti,
                                                               mode='max',
                                                               factor=factor,
                                                               patience=patience,
                                                               threshold=tol,
                                                               verbose=False)
        hist = train(meta_pred, pred, data, train_split, opti, sched, tol, early_stop, n_epoch, batch_size, crit, DEVICE)
        if plot in ['loss', 'acc']:
            plt.plot(hist['epoch'], hist[f'train_{plot}'], c='b', lw=2, alpha=train_split)
            plt.plot(hist['epoch'], hist[f'valid_{plot}'], c='b', lw=2, alpha=train_split, ls='--')
    if plot:
        plt.xlabel('Epoch')
        plt.ylabel(f'{plot}')
        plt.legend(['Train', 'Valid'])
        plt.show()
deeprm()