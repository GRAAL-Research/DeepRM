import torch

from src.model.predictor.predictor import Predictor


class LinearClassifier(Predictor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.params = torch.tensor([])
        self.n_features = config["n_features"]

    def set_params(self, params):
        self.params = params

    @property
    def n_params(self) -> int:
        n_bias = 1
        n_weights = self.n_features
        return n_bias + n_weights

    def forward(self, instances: torch.Tensor, **kwargs) -> tuple:
        bias_idx = -1
        target_dim = -1
        weights = self.params[:, :bias_idx]
        x = instances[:, :, :target_dim].transpose(0, 1)
        wx = (x * weights).sum(dim=-1)
        bias = self.params[:, bias_idx]
        output = wx + bias
        output = output.transpose(0, 1)

        return self._process_output(output)
