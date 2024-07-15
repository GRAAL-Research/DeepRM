import torch
from torch import nn as nn

from src.model.data_compressor.DataEncoder import DataEncoder
from src.model.mlp import MLP


class FSPool(nn.Module, DataEncoder):

    def __init__(self, config: dict, mlp_hidden_dims: list[int]) -> None:
        n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
        super(FSPool, self).__init__()
        self.output_dimension = mlp_hidden_dims[-1]

        self.mlp = MLP(config["n_features"], mlp_hidden_dims, config["device"], config["has_skip_connection"],
                       config["has_batch_norm"], "cnt", config["init_scheme"])

        self.mat = torch.rand((n_instances_per_class_per_dataset, self.output_dimension))
        if config["device"] == "gpu":
            self.mat = self.mat.to("cuda:0")

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        features = instances[:, :, :-1].clone()
        outputs = self.mlp.forward(features)
        outputs = torch.sort(outputs, dim=2)[0]
        targets = instances[:, :, -1].reshape((len(instances), -1, 1))

        outputs = torch.mean(outputs * self.mat * targets, dim=1)
        return outputs.reshape((len(outputs), -1))

    def get_output_dimension(self) -> int:
        return self.output_dimension
