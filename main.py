# Copyright 2023 Benjamin Leblanc, Mathieu Bazinet, Alexandre Drouin, Pascal Germain, Valentina Zantedeschi

# This file is part of the work Deep Reconstruction Machine (DeepRM).

# DeepRM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from train import *
from datasets import *
import torch

def deeprm(n = 1000,
           m = 20,
           d = 10,
           dataset = 'easy_gauss_lin',
           train_valid = 0.8,
           net = 'simplenet',
           kernel_dim = [200, 200],
           hidden_dim = [10, 200, 200],
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
           plot = True,
           DEVICE = "cpu"
    ):
    if dataset == 'easy_gauss_lin':
        data = data_gen(n, m, d, 'easy')
    elif dataset == 'hard_gauss_lin':
        data = data_gen(n, m, d, 'hard')
    if net == 'simplenet':
        network = SimpleNet(d, kernel_dim, hidden_dim, d+1, m, d, batch_size)
    if criterion == 'bce_loss':
        crit = BCELoss_mod
    if optimizer == 'adam':
        opti = torch.optim.Adam(network.parameters(), lr=copy.copy(start_lr))
    if scheduler == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti,
                                                               mode='max',
                                                               factor=factor,
                                                               patience=patience,
                                                               threshold=tol,
                                                               verbose=False)
    hist = train(network, data, train_valid, opti, sched, early_stop, n_epoch, batch_size, crit, DEVICE)
    if plot:
        plt.plot(hist['epoch'], hist['train'], c='b', lw=2, alpha=train_valid)
        plt.plot(hist['epoch'], hist['valid'], c='b', lw=2, alpha=train_valid, ls='--')
        plt.legend(['Train', 'Valid'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
deeprm()