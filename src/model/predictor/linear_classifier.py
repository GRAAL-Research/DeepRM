import torch

from src.model.predictor.predictor import Predictor


class LinearClassifier(Predictor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.weights = torch.tensor([])
        self.n_features = config["n_features"]

    def set_weights(self, weights):
        self.weights = weights

    @property
    def n_params(self) -> int:
        n_bias = 1
        n_weights = self.n_features
        return n_bias + n_weights

    def forward(self, instances: torch.Tensor, **kwargs) -> tuple:
        bias_idx = -1
        target_dim = -1
        w = self.weights[:, :bias_idx]
        x = instances[:, :, :target_dim].transpose(0, 1)
        wx = (x * w).sum(dim=-1)
        b = self.weights[:, bias_idx]
        output = wx + b
        output = output.transpose(0, 1)

        return self._process_output(output)
