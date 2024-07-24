import torch
from torch import nn as nn

from src.model.data_encoder.fspool import FSPool
from src.model.data_encoder.kme import KME
from src.model.mlp import MLP


class Attention(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Attention, self).__init__()
        self.temperature = config["attention_temperature"]
        self.keys = MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                        config["has_skip_connection"], config["has_batch_norm"], "cnt", config["init_scheme"])
        self.queries = self.create_queries(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.queries(x)
        k = self.keys(x)

        qkt = q.matmul(k.transpose(-2, -1))
        max_qkt_value = qkt.max(dim=-1).values
        scaled_qkt = self.temperature * qkt / max_qkt_value

        return torch.softmax(scaled_qkt, dim=-1)

    @staticmethod
    def create_queries(config: dict) -> nn.Module:
        if config["attention_pooling_type"].lower() == "kme":
            return KME(config, hidden_dims=config["attention_dim"])
        elif config["attention_pooling_type"].lower() == "fspool":
            return FSPool(config, config["attention_dim"])
        elif config["attention_pooling_type"].lower() == "none":
            return MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                       config["has_skip_connection"], config["has_batch_norm"], "cnt", config["init_scheme"])

        raise NotImplementedError(f"The pooling '{config['attention_pooling_type']}' is not supported.")
