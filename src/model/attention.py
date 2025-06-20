import torch
from torch import nn as nn

from src.model.data_encoder.deepset import DeepSet
from src.model.mlp import MLP


class Attention(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Attention, self).__init__()
        self.temperature = config["attention_temperature"]
        self.keys = MLP(config["n_features"] + config["target_size"], config["attention_dim"], config["device"],
                        config["has_skip_connection"], config["has_batch_norm"],
                        config["init_scheme"], "cnt")
        self.queries = self.create_queries(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.unsqueeze(self.queries.forward(x), 1)
        k = self.keys.forward(x)
        qkt = q.matmul(k.transpose(-2, -1))
        max_qkt_value = torch.unsqueeze(qkt.max(dim=-1).values, dim=2)
        scaled_qkt = self.temperature * qkt / max_qkt_value

        return torch.softmax(scaled_qkt, dim=-1)

    @staticmethod
    def create_queries(config: dict) -> nn.Module:
        if config["attention_pooling_type"] is None:
            return MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                       config["has_skip_connection"], config["has_batch_norm"], config["init_scheme"], "cnt")
        elif config["attention_pooling_type"].lower() == "deepset":
            return DeepSet(config, hidden_dims=config["attention_dim"])

        raise NotImplementedError(f"The pooling '{config['attention_pooling_type']}' is not supported.")
