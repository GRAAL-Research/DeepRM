import torch
from torch import nn as nn

from src.model.data_compressor.fspool import FSPool
from src.model.data_compressor.kme import KME
from src.model.mlp import MLP


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dims_mlp, hidden_dims_kme, n_instances_per_class_per_dataset: int, device: str,
                 init_scheme: str, has_skip_connection: bool, has_batch_norm: bool, pooling_type: str,
                 temperature: int) -> None:
        """
        Initialize a custom attention head.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims_mlp (list of int): architecture of the MLP;
            hidden_dims_kme (list of int): architecture of the embedding (MLP) in the KME;
            pooling_type (str): type of pooling to apply for the query computation
            temperature (float): temperature parameter for the softmax computation.
        """
        super(Attention, self).__init__()
        self.temperature = temperature
        #   The Keys are always computed by an MLP...
        self.k = MLP(input_dim, hidden_dims_mlp, device, has_skip_connection, has_batch_norm, "cnt", init_scheme)
        #   While the Queries might be the result of a pooling component.
        if pooling_type == "kme":
            self.q = KME(input_dim, hidden_dims_kme, device, init_scheme, has_skip_connection, has_batch_norm)
        elif pooling_type == "fspool":
            self.q = FSPool(input_dim, hidden_dims_kme, n_instances_per_class_per_dataset, device, init_scheme,
                            has_skip_connection, has_batch_norm)
        elif pooling_type == "none":
            self.q = MLP(input_dim, hidden_dims_kme, device, has_skip_connection, has_batch_norm, "cnt", init_scheme)
        else:
            raise NotImplementedError(f"The pooling '{pooling_type}' is not supported.")

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
        dist = torch.softmax(self.temperature * qkt / torch.max(qkt, dim=-1).values, dim=2)
        return dist
