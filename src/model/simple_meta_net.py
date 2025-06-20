import torch
import torch.nn.functional as F
from torch import nn as nn

from src.model.attention import Attention
from src.model.data_encoder.create_data_encoder import create_data_compressor
from src.model.mlp import MLP


class SimpleMetaNet(nn.Module):
    def __init__(self, config: dict, pred_n_params: int) -> None:
        """
        pred_input_dim (int): Input dimension of the predictor;
        """
        super().__init__()
        self.compression_set_size = config["compression_set_size"]
        self.compression_pool_size = config["compression_pool_size"]
        self.msg_type = config["msg_type"]
        self.msg_std = config["msg_std"]
        self.msg_size = config["msg_size"]
        self.meta_batch_size = config["meta_batch_size"]
        self.device = config["device"]
        self.is_using_a_random_msg = config["is_using_a_random_msg"]
        self.is_in_test_mode = False
        self.module_1_dim = config["module_1_dim"] + [config["msg_size"]]
        self.msg = torch.tensor(0.0)  # The default message
        self.msk = None  # Mask (compression selection)
        # The number of parameters the predictor has corresponds to the output dim of the meta-predictor.
        self.output_dim = pred_n_params
        # The first component of our meta-learner is a data compressor; a way to map a whole dataset to a vector.
        self.data_compressor_1 = create_data_compressor(config)
        module_1_input_dim = self.data_compressor_1.get_output_dimension()
        if self.compression_set_size > 0:
            # The module computing the message has the mapped compression set as input
            module_1_input_dim += config["deepset_dim"][-1]
        self.module_1 = MLP(module_1_input_dim, self.module_1_dim, config["device"],
                            config["has_skip_connection"], config["has_batch_norm"],
                            config["init_scheme"], config["msg_type"])
        # For each example in the copression wet, an attention head is required (see article).
        self.cas = nn.ModuleList([Attention(config) for _ in range(self.compression_set_size)])
        # That data compressor maps the compression set to a vectorial representation
        self.data_compressor_2 = create_data_compressor(config)
        # That module takes as input the mapped compression set and the message, outputs the downstream pred. params.
        self.module_2 = MLP(self.compute_module_2_input_dim(), config["module_2_dim"] + [self.output_dim],
                            config["device"], config["has_skip_connection"], config["has_batch_norm"],
                            config["init_scheme"])

    def get_message(self):
        return self.msg

    def compute_module_2_input_dim(self) -> int:
        mod_2_input_dim = 0

        if self.compression_set_size > 0:
            mod_2_input_dim += self.data_compressor_2.get_output_dimension()

        if self.msg_type is not None and self.msg_size > 0:
            mod_2_input_dim += self.module_1_dim[-1]

        if self.compression_set_size + self.msg_size == 0:
            mod_2_input_dim = 1

        return mod_2_input_dim

    def forward(self, x: torch.Tensor, n_noisy_messages: int = 0, is_in_test_mode: bool = False) -> torch.Tensor:
        curr_batch_size = x.shape[0]
        self.is_in_test_mode = is_in_test_mode
        msg_module_output = None
        compression_module_output = None

        # Compute the compression set, if necessary
        if self.compression_set_size > 0:
            compression_module_output = self.forward_compression_module(x, n_noisy_messages)

        # Compute that message, helped by the mapped compression set, if necessary
        if self.msg_size > 0:
            msg_module_output = self.forward_msg_module(x, compression_module_output, n_noisy_messages)
            self.msg = msg_module_output.clone()

        # Compute the dowstream predictor parameters, given a message and a mapped compressions set
        return self.forward_module_2(msg_module_output, compression_module_output, curr_batch_size)

    def forward_compression_module(self, x: torch.Tensor, n_noisy_messages: int) -> torch.Tensor:
        mask = self.cas[0].forward(x.clone())
        for j in range(1, len(self.cas)):
            out = self.cas[j].forward(x.clone())
            mask = torch.hstack((mask, out))
        # A 2D soft mask is computed
        if self.is_in_test_mode:
            # In test mode, the mask needs to be hard (one-hot vector).
            if self.compression_pool_size is not None:
                max_dim = mask.shape[2]
                mask[:, :, min(self.compression_pool_size, max_dim):] = 0
            mask = F.one_hot(torch.argmax(mask, dim=2, keepdim=False), num_classes=mask.shape[2]).type(torch.float)
        # The mask and the initial dataset are multiplied...
        x_masked = mask.matmul(x.clone())
        # ... Resulting in a compression set that needs to be mapped to a vector.
        x_masked = self.data_compressor_2.forward(x_masked)

        # Concatenating all the information (mask + msg)
        x_masked = torch.reshape(x_masked, (len(x_masked), -1))
        # Create copies of the mapped compression set if, later on, various noisy messages are to be appended.
        if n_noisy_messages > 0:
            x_masked = x_masked.repeat(n_noisy_messages, 1)

        return x_masked

    def forward_msg_module(self, x: torch.Tensor, compression_module_output: torch.Tensor,
                           n_noisy_messages: int) -> torch.Tensor:
        if self.is_using_a_random_msg:
            return self.create_random_message(x.shape[0])

        # The task is mapped to a vector...
        x = self.data_compressor_1.forward(x)
        # ... Then the mapped compression set is appended...
        if compression_module_output is not None:
            x = torch.hstack((x, compression_module_output[:x.shape[0], :]))
        # ... Then the message is computed.
        message = self.module_1.forward(x)

        if self.msg_type == "cnt":
            message = message * 3  # See bound computation

        if n_noisy_messages == 0:
            return message

        elif n_noisy_messages > 0:
            noisy_messages = [self.create_noisy_message(message) for _ in range(n_noisy_messages)]
            return torch.stack(noisy_messages).squeeze(1)

        raise ValueError(f"The number of noisy messages must be greater or equal to 0.")

    def create_random_message(self, meta_batch_size: int) -> torch.Tensor:
        random_message = torch.rand((meta_batch_size, self.msg_size))
        if self.device == "gpu":
            random_message = random_message.cuda()
        return random_message

    def create_noisy_message(self, x: torch.Tensor) -> torch.Tensor:
        # The underlying distribution is a standard isotropic Gaussian distribution.
        return torch.normal(x, self.msg_std)

    def forward_module_2(self, msg_module_output: torch.Tensor = None,
                         compression_module_output: torch.Tensor = None, curr_batch_size: int = 0) -> torch.Tensor:

        if msg_module_output is not None and compression_module_output is not None:
            merged_msg_and_compression_output = torch.cat((msg_module_output, compression_module_output),
                                                          dim=msg_module_output.ndim - 1)
            return self.module_2.forward(merged_msg_and_compression_output)

        elif msg_module_output is not None:
            return self.module_2.forward(msg_module_output)

        elif compression_module_output is not None:
            return self.module_2.forward(compression_module_output)

        else:
            if self.device == 'gpu':
                return self.module_2.forward(torch.zeros((curr_batch_size, 1)).cuda())
            else:
                return self.module_2.forward(torch.zeros((curr_batch_size, 1)))

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
