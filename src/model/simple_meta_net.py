import torch
from torch import nn as nn

from src.model.attention import Attention
from src.model.data_encoder.create_data_encoder import create_data_compressor_1
from src.model.data_encoder.kme import KME
from src.model.mlp import MLP


class SimpleMetaNet(nn.Module):
    def __init__(self, config: dict, pred_input_dim: int) -> None:
        """
        pred_input_dim (int): Input dimension of the predictor;
        """
        super().__init__()
        self.compression_set_size = config["compression_set_size"]
        self.msg_size = config["msg_size"]
        self.msg_type = config["msg_type"]
        self.n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
        self.mlp_1_dim = config["mlp_1_dim"] + [self.msg_size]
        self.msg = torch.tensor(0.0)
        self.msk = None  # Mask (compression selection)
        self.output_dim = pred_input_dim

        self.data_compressor_1 = create_data_compressor_1(config)
        self.module_1 = MLP(self.data_compressor_1.get_output_dimension(), self.mlp_1_dim, config["device"],
                            config["has_skip_connection"], config["has_batch_norm"], config["msg_type"],
                            config["init_scheme"])

        self.cas = nn.ModuleList([Attention(config) for _ in range(self.compression_set_size)])
        self.kme_2 = KME(config, hidden_dims=config["kme_dim"])

        self.module_2 = MLP(self.compute_module_2_input_dim(), config["mlp_2_dim"] + [self.output_dim], config["device"],
                            config["has_skip_connection"], config["has_batch_norm"], "none", config["init_scheme"])

    def compute_module_2_input_dim(self) -> int:
        mod_2_input_dim = 0

        if self.compression_set_size > 0:
            mod_2_input_dim += self.data_compressor_1.get_output_dimension()

        if self.msg_size > 0:
            mod_2_input_dim += self.mlp_1_dim[-1]

        return mod_2_input_dim

    def forward(self, x: torch.Tensor, n_random_messages: int = 0) -> torch.Tensor:
        """
        n_random_messages (int): number of random message to generate (0 to use mean as single message).
        """
        msg_module_output = None
        compression_module_output = None

        if self.msg_size > 0:
            msg_module_output = self.forward_msg_module(x, n_random_messages)

        if self.compression_set_size > 0:
            compression_module_output = self.forward_compression_module(x, n_random_messages)

        return self.forward_module_2(msg_module_output, compression_module_output)

    def forward_msg_module(self, x, n_random_messages):
        x = self.data_compressor_1.forward(x)
        x = self.module_1.forward(x)

        if self.msg_type == "cnt":
            x = x * 3  # See bound computation
        if n_random_messages == 0:
            self.msg = x.clone()
        if n_random_messages > 0:
            x_reshaped = x.unsqueeze(-1)
            for random_message_idx in range(n_random_messages):
                normal_dist = torch.normal(x_reshaped, 1).reshape((len(x), -1))
                if random_message_idx == 0:
                    self.msg = normal_dist
                else:
                    self.msg = torch.vstack((self.msg, normal_dist))
            x = self.msg

        return x

    def forward_compression_module(self, x, n_random_messages):
        mask = self.cas[0].forward(x.clone())
        for j in range(1, len(self.cas)):
            out = self.cas[j].forward(x.clone())
            mask = torch.hstack((mask, out))

        x_masked = mask.matmul(x.clone())
        x_masked = self.kme_2.forward(x_masked)

        # Concatenating all the information (mask + msg)
        x_masked = torch.reshape(x_masked, (len(x_masked), -1))
        if n_random_messages > 0:
            x_masked = x_masked.repeat(n_random_messages, 1)

        return x_masked

    def forward_module_2(self, msg_module_output: torch.Tensor | None,
                         compression_module_output: torch.Tensor | None) -> torch.Tensor:

        if msg_module_output is not None and compression_module_output is not None:
            merged_msg_and_compression_output = torch.hstack((msg_module_output, compression_module_output))
            return self.module_2.forward(merged_msg_and_compression_output)

        if msg_module_output is not None:
            return self.module_2.forward(msg_module_output)

        if compression_module_output is not None:
            return self.module_2.forward(compression_module_output)

        raise ValueError(f"The message module and the compression module are both disabled.")

    def compute_compression_set(self, x: torch.Tensor) -> None:
        """
        Targets the examples that have the most contributed in the compression set.
        """
        # Mask computation #
        if self.compression_set_size > 0:
            mask = self.cas[0](x)
            for j in range(1, len(self.cas)):
                out = self.cas[j](x)
                mask = torch.hstack((mask, out))
            self.msk = torch.topk(mask, 1, dim=2).indices.squeeze()
        else:
            raise ValueError("Cannot compute the compression set when it is of size 0.")
