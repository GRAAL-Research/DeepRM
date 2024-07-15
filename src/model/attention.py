import torch
from torch import nn as nn

from src.model.data_compressor.fspool import FSPool
from src.model.data_compressor.kme import KME
from src.model.mlp import MLP


class Attention(nn.Module):
    def __init__(self, config, pooling_type: str) -> None:
        """
        Args:
            pooling_type (str): type of pooling to apply for the query computation
        """
        super(Attention, self).__init__()
        self.temperature = config["attention_temperature"]

        #   The Keys are always computed by an MLP...
        self.keys = MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                        config["has_skip_connection"], config["has_batch_norm"], "cnt",
                        config["init_scheme"])
        #   While the Queries might be the result of a pooling component.
        if pooling_type == "kme":
            self.queries = KME(config, hidden_dims=config["attention_dim"])
        elif pooling_type == "fspool":
            self.queries = FSPool(config["n_features"] + 1, config["attention_dim"],
                                  config["n_instances_per_dataset"] // 2, config["device"],
                                  config["init_scheme"],
                                  config["has_skip_connection"], config["has_batch_norm"])
        elif pooling_type == "none":
            self.queries = MLP(config["n_features"] + 1, config["attention_dim"], config["device"],
                               config["has_skip_connection"], config["has_batch_norm"],
                               "cnt", config["init_scheme"])
        else:
            raise NotImplementedError(f"The pooling '{pooling_type}' is not supported.")

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
