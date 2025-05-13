import torch

from src.model.data_encoder.data_encoder import DataEncoder


class PassThroughModule(DataEncoder):

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.output_dim = config["n_features"]

    def forward(self, instances: torch.tensor) -> torch.tensor:
        return instances

    def get_output_dimension(self) -> int:
        return self.output_dim
