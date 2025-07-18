import torch

from src.model.data_encoder.data_encoder import DataEncoder
from src.model.mlp import MLP


class DeepSet(DataEncoder):
    """
    Corresponds to the DeepSet module presented in the article.
    """
    def __init__(self, config: dict, hidden_dims: list[int]) -> None:
        super().__init__()
        self.output_dim = hidden_dims[-1]
        self.mlp = MLP(config["n_features"], hidden_dims, config["device"], config["has_skip_connection"],
                       config["has_batch_norm"], config["init_scheme"])
        self.task = config["task"]
        self.target_size = config["target_size"]

    def forward(self, instances: torch.tensor) -> torch.tensor:
        features = instances[:, :, :-self.target_size]
        targets = instances[:, :, -self.target_size:]
        embeddings = self.mlp(features)
        target_size = targets.shape[-1]
        embedding_last_dim = embeddings.shape[-1]
        targets = targets.repeat(1, 1, embedding_last_dim // target_size)

        target_size = targets.shape[-1]
        embedding_last_dim = embeddings.shape[-1]
        targets = targets.repeat(1, 1, embedding_last_dim // target_size)
        return (embeddings * targets).mean(dim=1)

    def get_output_dimension(self) -> int:
        return self.output_dim
