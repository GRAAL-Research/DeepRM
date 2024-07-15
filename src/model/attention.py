import torch
from torch import nn as nn

from src.model.data_encoder.fspool import FSPool
from src.model.data_encoder.kme import KME
from src.model.mlp import MLP


class Attention(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Attention, self).__init__()
        self.temperature = config["attention_temperature"]

        #   The Keys are always computed by an MLP...
        self.keys = MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                        config["has_skip_connection"], config["has_batch_norm"], "cnt", config["init_scheme"])
        #   While the Queries might be the result of a pooling component.

        if config["attention_pooling_type"].lower() == "kme":
            self.queries = KME(config, hidden_dims=config["attention_dim"])
        elif config["attention_pooling_type"].lower() == "fspool":
            self.queries = FSPool(config, config["attention_dim"])
        elif config["attention_pooling_type"].lower() == "none":
            self.queries = MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                               config["has_skip_connection"], config["has_batch_norm"], "cnt", config["init_scheme"])
        else:
            raise NotImplementedError(f"The pooling '{config['attention_pooling_type']}' is not supported.")

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1, x_2 = x.clone(), x.clone()
        queries = self.queries.forward(x_1)
        keys = self.keys.forward(x_2)
        qkt = torch.matmul(queries, torch.transpose(keys, 1, 2))
        dist = torch.softmax(self.temperature * qkt / torch.max(qkt, dim=-1).values, dim=2)
        return dist
