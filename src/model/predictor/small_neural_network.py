import torch
from torch import nn

from src.model.lazy_batch_norm import LazyBatchNorm
from src.model.mlp import MLP
from src.model.predictor.predictor import Predictor


class SmallNeuralNetwork(Predictor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.architecture_sizes = [config["n_features"]] + config["pred_hidden_sizes"] + [config["target_size"]]
        self.mlp = MLP(config["n_features"], self.architecture_sizes[1:], config["device"],
                       config["has_skip_connection"], config["has_batch_norm"], config["batch_norm_min_dim"])

        self.params = torch.tensor([])
        self.batch_norm_params = []
        self.use_last_values = False
        self.save_bn_params = False
        self.target_size = config["target_size"]

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_params(self, params: torch.Tensor) -> None:
        self.params = params

    def set_forward_mode(self, use_last_values: bool = False, save_bn_params: bool = False):
        self.use_last_values = use_last_values
        self.save_bn_params = save_bn_params

    def reset_forward_mode(self):
        self.use_last_values = False
        self.save_bn_params = False

    def get_batch_norm_from_prior(self, prior):
        for i in range(len(prior.mlp)):
            if isinstance(prior.mlp[i], LazyBatchNorm):
                self.mlp.mlp[i] = prior.mlp[i]

    def forward(self, instances: torch.Tensor) -> tuple:
        x = instances[:, :, :-self.target_size]
        batch_size = len(x)

        if not self.use_last_values:
            self.batch_norm_params = []

        params_low_idx = 0
        params_high_idx = 0
        linear_layer_idx = 0

        for layer in self.mlp.get_modules():
            if isinstance(layer, nn.Linear):
                current_linear_layer_dim = self.architecture_sizes[linear_layer_idx]
                new_linear_layer_dim = self.architecture_sizes[linear_layer_idx + 1]

                params_high_idx += current_linear_layer_dim * new_linear_layer_dim
                weights = self.params[:, params_low_idx: params_high_idx].reshape(batch_size,
                                                                                  new_linear_layer_dim,
                                                                                  current_linear_layer_dim)

                params_low_idx += current_linear_layer_dim * new_linear_layer_dim
                params_high_idx += new_linear_layer_dim
                bias = self.params[:, params_low_idx: params_high_idx].unsqueeze(dim=-2)

                weights = weights.transpose(-2, -1)
                x = x.matmul(weights) + bias

                params_low_idx += new_linear_layer_dim
                linear_layer_idx += 1

            elif isinstance(layer, LazyBatchNorm):
                if self.use_last_values:
                    x = layer.forward(x, is_using_saved_stats=self.use_last_values)
                elif self.save_bn_params:
                    x = layer.forward(x, is_saving_computed_stats=self.save_bn_params)
                else:
                    x = layer.forward(x)

            else:
                x = layer(x)
        return self._process_output(x)
