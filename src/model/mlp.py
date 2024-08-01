from typing import Final

import torch
import torch.nn.functional as F
from torch import nn as nn

from src.model.lazy_batch_norm import LazyBatchNorm
from src.model.utils.initialize_weights import initialize_weights
from src.model.utils.sign_straight_through import SignStraightThrough


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], device: str, has_skip_connection: bool,
                 has_batch_norm: bool, batch_norm_min_dim: int, init_scheme: str = None,
                 msg_type: str | None = None, has_msg_as_input: bool = False) -> None:
        super(MLP, self).__init__()
        self.has_skip_connection = has_skip_connection

        input_and_hidden_dims = MLP.compute_input_and_hidden_dims(input_dim, hidden_dims)
        self.mlp = MLP.create_mlp(has_batch_norm, batch_norm_min_dim, msg_type,
                                  has_msg_as_input, input_and_hidden_dims, device)

        last_layer_idx = len(self.mlp) - 1
        self.skip_position = last_layer_idx - has_batch_norm

        if device == "gpu":
            self.mlp.cuda()

        if init_scheme:
            initialize_weights(init_scheme, self.mlp)

    def get_modules(self):
        return self.mlp

    @staticmethod
    def compute_input_and_hidden_dims(input_dim: int, hidden_dims: list[int]) -> list[int]:
        input_and_hidden_dims = [input_dim] + hidden_dims

        return input_and_hidden_dims

    @staticmethod
    def create_mlp(has_batch_norm: bool, batch_norm_min_dim: int, msg_type: str, has_msg_as_input: bool,
                   input_and_hidden_dims: list[int], device: str = "cpu") -> nn.ModuleList:
        modules = torch.nn.ModuleList()
        for dim_idx in range(len(input_and_hidden_dims) - 1):
            if (has_batch_norm and batch_norm_min_dim <= input_and_hidden_dims[dim_idx] and
                    (not has_msg_as_input or dim_idx != 0)):
                modules.append(LazyBatchNorm(device))
            modules.append(nn.Linear(input_and_hidden_dims[dim_idx], input_and_hidden_dims[dim_idx + 1]))

            is_last_layer = dim_idx == len(input_and_hidden_dims) - 2
            activation_function = MLP.create_activation_function(is_last_layer, msg_type)
            modules.append(activation_function)

        return modules

    @staticmethod
    def create_activation_function(is_last_layer: bool, msg_type: str | None) -> nn.Module:
        if not is_last_layer:
            return nn.ReLU()
        elif msg_type is None:
            return nn.Identity()
        elif msg_type == "dsc":
            return SignStraightThrough()
        elif msg_type == "cnt":
            return nn.Tanh()

        raise NotImplementedError(f"The message type '{msg_type}' is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_x = x.clone()
        for layer_idx, layer in enumerate(self.mlp, start=1):
            if self.has_skip_connection and layer_idx == self.skip_position:
                x = self.apply_skip_connection(x, initial_x)
            x = layer(x)

        return x

    @staticmethod
    def apply_skip_connection(x: torch.Tensor, initial_x: torch.Tensor) -> torch.tensor:
        padding_size = x.shape[-1] - initial_x.shape[-1]
        padding = F.pad(input=initial_x, pad=(0, padding_size, 0, 0), mode="constant", value=0)
        return x + padding
