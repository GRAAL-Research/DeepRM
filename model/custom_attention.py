import torch
from torch import nn as nn

from model.fspool import FSPool
from model.kme import KME
from utils import MLP


class CA(nn.Module):
    def __init__(self, input_dim, hidden_dims_mlp, hidden_dims_kme, m, device, init, skip, bn, pool, temp):
        """
        Initialize a custom attention head.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims_mlp (list of int): architecture of the MLP;
            hidden_dims_kme (list of int): architecture of the embedding (MLP) in the KME;
            m (int): number of examples per dataset;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not;
            pool (str): type of pooling to apply for the query computation (choices: 'kme', 'fspool', 'none');
            temp (float): temperature parameter for the softmax computation.
        """
        super(CA, self).__init__()
        self.temp = temp
        #   The Keys are always computed by an MLP...
        self.k = MLP(input_dim, hidden_dims_mlp, device, init, skip, bn, 'cnt')
        #   While the Queries might be the result of a pooling component.
        if pool == 'kme':
            self.q = KME(input_dim, hidden_dims_kme, device, init, skip, bn)
        elif pool == 'fspool':
            self.q = FSPool(input_dim, hidden_dims_kme, m, device, init, skip, bn)
        elif pool == 'none':
            self.q = MLP(input_dim, hidden_dims_kme, device, init, skip, bn, 'cnt')
        else:
            assert False, 'Wrong pooling choice.'

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1, x_2 = x.clone(), x.clone()
        queries = self.q.forward(x_1)
        keys = self.k.forward(x_2)
        qkt = torch.matmul(queries, torch.transpose(keys, 1, 2))
        dist = torch.softmax(self.temp * qkt / torch.max(qkt, dim=-1).values, dim=2)
        return dist
