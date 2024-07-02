import torch
from torch import nn as nn

from utils import MLP


class KME(nn.Module):
    def __init__(self, input_dim, hidden_dims, device, init, skip:bool = False, bn : bool = False):
        """
        Initialize a custom attention head.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the embedding;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not.
        """
        super(KME, self).__init__()
        self.embedding = MLP(input_dim - 1, hidden_dims, device, init, skip, bn, 'none')

    def forward(self, x):
        """
        Computes the KME output, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1 = x[:, :, :-1].clone()
        out = self.embedding.forward(x_1)
        x_1 = torch.mean(out * torch.reshape(x[:, :, -1], (len(x), -1, 1)), dim=1)  # ... A compression is done
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))
