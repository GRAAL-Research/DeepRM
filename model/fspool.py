import torch
from torch import nn as nn

from utils import MLP


class FSPool(nn.Module):
    def __init__(self, input_dim, hidden_dims, m, device, init, skip, bn):
        """
        Initialize .
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the embedding;
            m (int): number of examples per dataset;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not.
        """
        super(FSPool, self).__init__()
        self.mlp = MLP(input_dim - 1, hidden_dims, device, init, skip, bn, 'cnt')
        self.mat = torch.rand((m, hidden_dims[-1]))
        if device == 'gpu':
            self.mat = self.mat.to('cuda:0')

    def forward(self, x):
        """
        Computes FSPool output, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1 = x[:, :, :-1].clone()
        out = self.mlp.forward(x_1)
        out = torch.sort(out, dim=2)[0]

        x_1 = torch.mean(out * self.mat * torch.reshape(x[:, :, -1], (len(x), -1, 1)), dim=1)
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))
