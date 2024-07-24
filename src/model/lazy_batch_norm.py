import torch
from torch import nn as nn


class LazyBatchNorm(nn.Module):
    def __init__(self, device: str = "cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.used = False
        self.gamma = None
        self.beta = None
        self.last_mean = None
        self.last_std = None
        self.last_gamma = None
        self.last_beta = None

    def forward(self, x, use_last_values=False, save_bn_params=False):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
            use_last_values (bool): whether to use previously saved means and stds;
            save_bn_params (bool): whether to save the computed means and stda for future use with same parameters;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        if not self.used:   # For the first usage, the dims are stored; scale and offsets are initialized
            shape = [1] * (len(x.shape) - 1) + [x.shape[-1]]
            self.gamma = torch.normal(1, torch.ones(shape) * 1e-5)
            self.beta = torch.normal(0, torch.ones(shape) * 1e-5)
            if self.device == "gpu":
                self.gamma = self.gamma.to(device='cuda:0')
                self.beta = self.beta.to(device='cuda:0')
            self.used = True
        if use_last_values:     # Whether to use previously saved means and stds
            return (x - self.last_mean) / (self.last_std + 1e-10) * \
                self.last_gamma + self.last_beta
        else:
            if save_bn_params:  # Whether to save the computed means and stds for future use with same parameters
                self.last_mean = x.mean(dim=1, keepdim=True)
                self.last_std = x.std(dim=1, keepdim=True)
                self.last_gamma = self.gamma.clone()
                self.last_beta = self.beta.clone()
                return ((x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-10) *
                        self.gamma + self.beta)
            return ((x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-10) *
                    self.gamma + self.beta)
