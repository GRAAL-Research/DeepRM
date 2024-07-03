import torch
from torch import nn as nn

from src.model.utils.initialize_weights import initialize_weights
from src.model.utils.sign_straight_through import SignStraightThrough


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, device: str, has_skip_connection: bool, has_batch_norm: bool,
                 msg_type: str, init_scheme: str = None) -> None:
        """
        Creates a ReLU linear layer, given dimensions.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the MLP;
        """
        super(MLP, self).__init__()
        self.dims = [input_dim] + hidden_dims
        self.has_skip_connection = has_skip_connection
        self.module = torch.nn.ModuleList()
        if self.has_skip_connection and len(self.dims) > 2:
            self.dims.insert(len(self.dims) - 1, self.dims[0])
        for k in range(len(self.dims) - 1):
            if has_batch_norm:
                self.module.append(nn.LazyBatchNorm1d())
            self.module.append(nn.Linear(self.dims[k], self.dims[k + 1]))

            if k < len(self.dims) - 2 or msg_type == "pos":
                self.module.append(nn.ReLU())
            elif k == len(self.dims) - 2 and msg_type == "none":
                self.module.append(nn.Identity())
            elif k == len(self.dims) - 2 and msg_type == "dsc":
                self.module.append(SignStraightThrough())
            elif k == len(self.dims) - 2 and msg_type == "cnt":
                self.module.append(nn.Tanh())
            else:
                raise NotImplementedError(f"The message type '{msg_type}' is not supported.")

        self.skip_position = len(self.module) - (1 + 1 * has_batch_norm)
        if device == "gpu":
            self.module.to("cuda:0")
        if init_scheme is not None:
            initialize_weights(init_scheme, self.module)

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        lay_cnt = 0
        for layer in self.module:
            lay_cnt += 1
            x_1 = x.clone()
            if self.has_skip_connection and lay_cnt == self.skip_position:
                x += x_1
            x = layer(x).clone()
        return x
