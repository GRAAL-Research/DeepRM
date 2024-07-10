import torch
from torch import nn as nn

from src.model.mlp import MLP


class KME(nn.Module):
    def __init__(self, input_dim, hidden_dims, device: str, init_scheme: str, has_skip_connection: bool,
                 has_batch_norm: bool, task: str) -> None:
        """
        hidden_dims (list of int): architecture of the embedding;
        """
        super(KME, self).__init__()
        self.embedding = MLP(input_dim, hidden_dims, device, has_skip_connection, has_batch_norm, 'none',
                             init_scheme)
        self.task = task

    def forward(self, x):
        """
        Computes the KME output, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1 = x.clone()
        out = self.embedding.forward(x_1)
        label = x[:, :, -1] if self.task == "classification" else x[:, :, -1] + 1
        return torch.mean(out * torch.reshape(label, (len(x), -1, 1)), dim=1)
