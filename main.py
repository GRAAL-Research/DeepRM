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
           d = 5,
           dataset = 'gauss_lin',
           net = 'simplenet',
           criterion = 'bce_loss',
           start_lr=1e-5,
           factor = 0.5,
           patience = 5,
           tol = 1e-4,
           optimizer = 'adam',
           scheduler = 'plateau',
           n_epoch = 200,
           batch_size = 10,
           DEVICE = "cpu"
    ):
    if dataset == 'gauss_lin':
        dataset = data_gen(n, m, d)
    if net == 'simplenet':
        Net = SimpleNet(2 * m * d, [20,20], d+1)
    if criterion == 'bce_loss':
        criterion = BCELoss_mod
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(Net.parameters(), lr=start_lr)
    if scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       factor=factor,
                                                       patience=patience,
                                                       threshold=tol,
                                                       verbose=False)
    train(Net, dataset, m, optimizer, scheduler, n_epoch, batch_size, criterion, DEVICE)

deeprm()