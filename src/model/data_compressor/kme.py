import torch
from torch import nn as nn

from src.model.data_compressor.DataEncoder import DataEncoder
from src.model.mlp import MLP


class KME(nn.Module, DataEncoder):

    def __init__(self, input_dim, hidden_dims, device: str, init_scheme: str, has_skip_connection: bool,
                 has_batch_norm: bool) -> None:
        """
        hidden_dims (list of int): architecture of the embedding;
        """
        super(KME, self).__init__()
        self.output_dim = hidden_dims[-1]

        self.embedding = MLP(input_dim - 1, hidden_dims, device, has_skip_connection, has_batch_norm, 'none',
                             init_scheme)

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
        return torch.mean(out * torch.reshape(x[:, :, -1], (len(x), -1, 1)), dim=1)

    def get_output_dimension(self) -> int:
        return self.output_dim
