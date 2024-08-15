import torch
from torch import nn as nn
import torch.nn.functional as F

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
        self.msg_type = config["msg_type"]
        self.msg_size = config["msg_size"]
        self.batch_size = config["batch_size"]
        self.device = config["device"]
        self.is_using_a_random_msg = config["is_using_a_random_msg"]
        self.test = None

        self.n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
        self.module_1_dim = config["module_1_dim"] + [config["msg_size"]]
        self.msg = torch.tensor(0.0)
        self.msk = None  # Mask (compression selection)
        self.output_dim = pred_input_dim

        self.data_compressor_1 = create_data_compressor_1(config)
        self.module_1 = MLP(self.data_compressor_1.get_output_dimension(), self.module_1_dim, config["device"],
                            config["has_skip_connection"], config["has_batch_norm"], config["batch_norm_min_dim"],
                            config["init_scheme"], config["msg_type"])

        self.cas = nn.ModuleList([Attention(config) for _ in range(self.compression_set_size)])
        self.kme_2 = KME(config, hidden_dims=config["kme_dim"])

        self.module_2 = MLP(self.compute_module_2_input_dim(), config["module_2_dim"] + [self.output_dim],
                            config["device"], config["has_skip_connection"], config["has_batch_norm"],
                            config["batch_norm_min_dim"], config["init_scheme"], has_msg_as_input=True)

    def get_message(self):
        return self.msg

    def compute_module_2_input_dim(self) -> int:
        mod_2_input_dim = 0

        if self.compression_set_size > 0:
            mod_2_input_dim += self.kme_2.get_output_dimension()

        if self.msg_type is not None and self.msg_size > 0:
            mod_2_input_dim += self.module_1_dim[-1]

        return mod_2_input_dim

    def forward(self, x: torch.Tensor, n_noisy_messages: int = 0, test: bool = False) -> torch.Tensor:
        self.test = test
        msg_module_output = None
        compression_module_output = None

        if self.msg_type is not None and self.msg_size > 0:
            msg_module_output = self.forward_msg_module(x, n_noisy_messages)
            self.msg = msg_module_output.clone()

        if self.compression_set_size > 0:
            compression_module_output = self.forward_compression_module(x, n_noisy_messages)

        return self.forward_module_2(msg_module_output, compression_module_output, n_noisy_messages)

    def forward_msg_module(self, x: torch.Tensor, n_noisy_messages: int) -> torch.Tensor:
        if self.is_using_a_random_msg:
            return self.create_random_message(x.shape[0])

        x = self.data_compressor_1.forward(x)
        message = self.module_1.forward(x)

        if self.msg_type == "cnt":
            message = message * 3  # See bound computation

        if n_noisy_messages == 0:
            return message

        elif n_noisy_messages > 0:
            noisy_messages = [self.create_noisy_message(message) for _ in range(n_noisy_messages)]
            return torch.stack(noisy_messages).squeeze(0)

        raise ValueError(f"The number of noisy messages must be greater or equal to 0.")

    def create_random_message(self, batch_size: int) -> torch.Tensor:
        random_message = torch.rand((batch_size, self.msg_size))
        if self.device == "gpu":
            random_message = random_message.cuda()
        return random_message

    @staticmethod
    def create_noisy_message(x: torch.Tensor) -> torch.Tensor:
        return torch.normal(x, 1)

    def forward_compression_module(self, x: torch.Tensor, n_noisy_messages: int) -> torch.Tensor:
        mask = self.cas[0].forward(x.clone())
        for j in range(1, len(self.cas)):
            out = self.cas[j].forward(x.clone())
            mask = torch.hstack((mask, out))
        if self.test:
            mask = F.one_hot(torch.argmax(mask, dim=2, keepdim=False), num_classes=mask.shape[2]).type(torch.float)
        x_masked = mask.matmul(x.clone())
        x_masked = self.kme_2.forward(x_masked)

        # Concatenating all the information (mask + msg)
        x_masked = torch.reshape(x_masked, (len(x_masked), -1))
        if n_noisy_messages > 0:
            x_masked = x_masked.repeat(n_noisy_messages, 1)

        return x_masked

    def forward_module_2(self, msg_module_output: torch.Tensor = None,
                         compression_module_output: torch.Tensor = None,
                         n_noisy_messages: int = 0) -> torch.Tensor:

        if msg_module_output is not None and compression_module_output is not None:
            if len(msg_module_output.shape) > len(compression_module_output.shape):
                compression_module_output = torch.unsqueeze(compression_module_output, dim=1)
            merged_msg_and_compression_output = torch.cat((msg_module_output, compression_module_output),
                                                          dim=msg_module_output.ndim - 1)
            return self.module_2.forward(merged_msg_and_compression_output)

        if msg_module_output is not None:
            return self.module_2.forward(msg_module_output)

        if compression_module_output is not None:
            if n_noisy_messages > 0:
                compression_module_output = torch.unsqueeze(compression_module_output, dim=1)
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
