import torch
from torch import nn as nn

from src.model.utils.initialize_weights import initialize_weights
from src.model.utils.sign_straight_through import SignStraightThrough


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], device: str, has_skip_connection: bool,
                 has_batch_norm: bool, msg_type: str, init_scheme: str = None) -> None:
        super(MLP, self).__init__()
        self.has_skip_connection = has_skip_connection

        input_and_hidden_dims = MLP.compute_input_and_hidden_dims(input_dim, hidden_dims, has_skip_connection)
        self.module = MLP.create_mlp(has_batch_norm, msg_type, input_and_hidden_dims)
        self.skip_position = len(self.module) - (1 + 1 * has_batch_norm)

        if device == "gpu":
            self.module.to("cuda:0")

        if init_scheme:
            initialize_weights(init_scheme, self.module)

    @staticmethod
    def compute_input_and_hidden_dims(input_dim: int, hidden_dims: list[int], has_skip_connection: bool) -> list[int]:
        input_and_hidden_dims = [input_dim] + hidden_dims
        # TODO fix the skip connection
        # if has_skip_connection and len(input_and_hidden_dims) > 2:
        #     input_and_hidden_dims.insert(len(input_and_hidden_dims) - 1, input_and_hidden_dims[0])

        return input_and_hidden_dims

    @staticmethod
    def create_mlp(has_batch_norm: bool, msg_type: str, input_and_hidden_dims: list[int]) -> nn.ModuleList:
        modules = torch.nn.ModuleList()
        for dim_idx in range(len(input_and_hidden_dims) - 1):
            if has_batch_norm:
                modules.append(nn.LazyBatchNorm1d())
            modules.append(nn.Linear(input_and_hidden_dims[dim_idx], input_and_hidden_dims[dim_idx + 1]))

            if dim_idx == len(input_and_hidden_dims) - 2 or msg_type == "pos":
                activation_function = MLP.create_activation_function(msg_type)
                modules.append(activation_function)

        return modules

    @staticmethod
    def create_activation_function(msg_type: str) -> nn.Module:
        if msg_type == "pos":
            return nn.ReLU()
        if msg_type == "none":
            return nn.Identity()
        if msg_type == "dsc":
            return SignStraightThrough()
        if msg_type == "cnt":
            return nn.Tanh()

        raise NotImplementedError(f"The message type '{msg_type}' is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.module, start=1):
            x_1 = x.clone()
            if self.has_skip_connection and idx == self.skip_position:
                x += x_1
            x = layer(x).clone()

        return x
