import torch
from IPython import embed
from torch.nn.functional import embedding

from src.model.data_encoder.data_encoder import DataEncoder
from src.model.mlp import MLP


class KME(DataEncoder):

    def __init__(self, config: dict, hidden_dims: list[int]) -> None:
        super().__init__()
        self.output_dim = hidden_dims[-1]
        self.mlp = MLP(config["n_features"], hidden_dims, config["device"], config["has_skip_connection"],
                       config["has_batch_norm"], config["batch_norm_min_dim"], config["init_scheme"])
        self.task = config["task"]
        self.target_size = config["target_size"]

    def forward(self, instances: torch.tensor) -> torch.tensor:
        features = instances[:, :, :-self.target_size]
        targets = instances[:, :, -self.target_size:]
        embeddings = self.mlp(features)

        targets = targets.repeat(1, 1, embeddings.shape[2] // targets.shape[2])
        return (embeddings * targets).mean(dim=1)

    def get_output_dimension(self) -> int:
        return self.output_dim
