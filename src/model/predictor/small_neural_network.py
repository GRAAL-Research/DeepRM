import torch
from torch import nn

from src.model.lazy_batch_norm import LazyBatchNorm
from src.model.mlp import MLP
from src.model.predictor.predictor import Predictor


class SmallNeuralNetwork(Predictor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        input_layer_size = config["n_features"]
        output_layer_size = config["label_size"]
        self.architecture_sizes = [input_layer_size] + config["pred_hidden_sizes"] + [output_layer_size]
        self.mlp = MLP(config["n_features"], self.architecture_sizes[1:], config["device"],
                       config["has_skip_connection"],
                       config["has_batch_norm"], "none")

        self.weights = torch.tensor([])
        self.batch_norm_params = []
        self.use_last_values = False
        self.save_bn_params = False

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_weights(self, weights: torch.Tensor) -> None:
        self.weights = weights

    def set_forward_mode(self, use_last_values: bool = False, save_bn_params: bool = False):
        self.use_last_values = use_last_values
        self.save_bn_params = save_bn_params

    def reset_forward_mode(self):
        self.use_last_values = False
        self.save_bn_params = False

    def forward(self, instances: torch.Tensor) -> tuple:
        if not self.use_last_values:
            self.batch_norm_params = []

        batch_size = len(instances)
        input_0 = instances[:, :, :-1]
        count_1 = 0
        count_2 = 0
        n_batch_norm = 0
        j = 0
        for layer in self.mlp.module:
            if isinstance(layer, nn.Linear):
                count_2 += self.architecture_sizes[j] * self.architecture_sizes[j + 1]
                w = torch.reshape(self.weights[:, count_1:count_2],
                                  (batch_size, self.architecture_sizes[j + 1], self.architecture_sizes[j]))
                count_1 += self.architecture_sizes[j] * self.architecture_sizes[j + 1]
                count_2 += self.architecture_sizes[j + 1]
                b = torch.reshape(self.weights[:, count_1:count_2], (batch_size, 1, self.architecture_sizes[j + 1]))
                count_1 += self.architecture_sizes[j + 1]
                j += 1
                w = torch.transpose(w, 1, 2)
                input_0 = torch.matmul(input_0, w) + b
            elif isinstance(layer, LazyBatchNorm):
                if self.use_last_values:
                    input_0 = layer.forward(input_0, use_last_values=self.use_last_values)
                elif self.save_bn_params:
                    input_0 = layer.forward(input_0, save_bn_params=self.save_bn_params)
                else:
                    input_0 = layer.forward(input_0)
                n_batch_norm += 1
            else:
                input_0 = layer(input_0)
        output = input_0
        output = output.transpose(1, 2).squeeze(dim=-2)

        return self._process_output(output)
