import torch

from src.model.data_encoder.data_encoder import DataEncoder
from src.model.mlp import MLP


class FSPool(DataEncoder):
    def __init__(self, config: dict, mlp_hidden_dims: list[int]) -> None:
        n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
        super().__init__()
        self.output_dimension = mlp_hidden_dims[-1]

        self.mlp = MLP(config["n_features"], mlp_hidden_dims, config["device"], config["has_skip_connection"],
                       config["has_batch_norm"], "cnt", config["init_scheme"])

        self.matrix = torch.rand((n_instances_per_class_per_dataset, self.output_dimension))
        if config["device"] == "gpu":
            self.matrix = self.matrix.to("cuda:0")

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        features = instances[:, :, :-1]
        outputs = self.mlp(features)
        outputs = torch.sort(outputs, dim=2)[0]
        targets = instances[:, :, -1].unsqueeze(-1)

        outputs = (outputs * self.matrix * targets).mean(dim=1)
        return outputs.squeeze(-1)

    def get_output_dimension(self) -> int:
        return self.output_dimension
