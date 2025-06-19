import torch
from torch import nn as nn


class LazyBatchNorm(nn.Module):
    """
    A custom batch norm layer, which is defined the first time it is used by the size of its input.
    """
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        self.gamma = None
        self.beta = None
        self.last_mean = None
        self.last_std = None
        self.last_gamma = None
        self.last_beta = None
        self.epsilon = 1e-10

    def initialize_gamma_and_beta(self, x_shape: torch.Size) -> None:
        params_shape = [1] * (len(x_shape) - 1) + [x_shape[-1]]
        epsilon = 1e-5
        std = torch.ones(params_shape) * epsilon
        self.gamma = nn.Parameter(torch.normal(mean=1, std=std))
        self.beta = nn.Parameter(torch.normal(mean=0, std=std))

        if self.device == "gpu":
            self.cuda()

    def forward(self, x: torch.Tensor, is_using_saved_stats: bool = False,
                is_saving_computed_stats: bool = False) -> torch.Tensor:
        if self.gamma is None and self.beta is None:
            self.initialize_gamma_and_beta(x.shape)

        if is_using_saved_stats:
            return (x - self.last_mean) / (self.last_std + self.epsilon) * self.last_gamma + self.last_beta

        new_mean = x.mean(dim=1, keepdim=True)
        new_std = self.compute_safe_std(x, dim=1)
        if is_saving_computed_stats:
            self.last_mean = new_mean.clone()
            self.last_std = new_std.clone()
            self.last_gamma = self.gamma.clone()
            self.last_beta = self.beta.clone()

        return (x - new_mean) / (new_std + self.epsilon) * self.gamma + self.beta

    def compute_safe_std(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[dim] == 1:
            standard_deviation = torch.zeros(x.size())
            if self.device == "gpu":
                standard_deviation = standard_deviation.cuda()
            return standard_deviation

        return x.std(dim=dim, keepdim=True)
