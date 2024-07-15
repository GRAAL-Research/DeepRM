import torch
from torch import nn as nn

from src.model.attention import Attention
from src.model.data_compressor.create_data_compressor import create_data_compressor_1
from src.model.data_compressor.kme import KME
from src.model.mlp import MLP


class SimpleMetaNet(nn.Module):
    def __init__(self, config: dict, pred_input_dim: int) -> None:
        """
        Generates the DeepRM meta-predictor.
        pred_input_dim (int): Input dimension of the predictor;
        """
        super(SimpleMetaNet, self).__init__()
        self.compression_set_size = config["compression_set_size"]
        self.msg_size = config["msg_size"]
        self.msg_type = config["msg_type"]
        self.n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
        self.mlp_1_dim = config["mlp_1_dim"] + [self.msg_size]
        self.msg = torch.tensor(0.0)  # Message (compression selection)
        self.msk = None  # Mask (compression selection)
        self.input_dim = config["n_features"]
        self.output_dim = pred_input_dim

        # Generating the many components (custom attention (CA) multi-heads, KME #1-2, MLP #1-2) of the meta-learner
        self.data_compressor_1 = create_data_compressor_1(config)
        self.mod_1 = MLP(self.data_compressor_1.get_output_dimension(), self.mlp_1_dim, config["device"],
                         config["has_skip_connection"], config["has_batch_norm"], config["msg_type"],
                         config["init_scheme"])

        self.cas = nn.ModuleList([])
        for i in range(self.compression_set_size):
            self.cas.append(Attention(config))

        self.kme_2 = KME(config, hidden_dims=config["kme_dim"])

        self.mod_2 = MLP(self.compute_mod_2_input_dim(), config["mlp_2_dim"] + [self.output_dim], config["device"],
                         config["has_skip_connection"], config["has_batch_norm"], "none", config["init_scheme"])

    def compute_mod_2_input_dim(self) -> int:
        mod_2_input_dim = 0

        if self.compression_set_size > 0:
            mod_2_input_dim += self.data_compressor_1.get_output_dimension()

        if self.msg_size > 0:
            mod_2_input_dim += self.mlp_1_dim[-1]

        return mod_2_input_dim

    def forward(self, x, n_samples=0):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
            n_samples (int): number of random message to generate (0 to use mean as single message).
        return:
            torch.Tensor: output of the network.
        """
        # Message computation #
        x_ori = x.clone()
        if self.msg_size > 0:
            x = self.data_compressor_1.forward(x)
            # Passing through MLP #1 #
            x = self.mod_1.forward(x)

            if self.msg_type == "cnt":
                x = x * 3  # See bound computation
            if n_samples == 0:
                self.msg = x.clone()
            if n_samples > 0:
                x_reshaped = torch.reshape(x, (-1, 1))
                for sample in range(n_samples):
                    if sample == 0:
                        self.msg = torch.reshape(torch.normal(x_reshaped, 1), (len(x), -1))
                    else:
                        self.msg = torch.vstack((self.msg, torch.reshape(torch.normal(x_reshaped, 1), (len(x), -1))))
                x = self.msg

        # Mask computation
        if self.compression_set_size > 0:
            mask = self.cas[0].forward(x_ori.clone())
            for j in range(1, len(self.cas)):
                out = self.cas[j].forward(x_ori.clone())
                mask = torch.hstack((mask, out))

            # Applying the mask to x #
            x_masked = torch.matmul(mask, x_ori.clone())

            # Passing through KME #1 #
            x_masked = self.kme_2.forward(x_masked)

            # Concatenating all the information (mask + msg) #
            x_masked = torch.reshape(x_masked, (len(x_masked), -1))
            if n_samples > 0:
                x_masked = x_masked.repeat(n_samples, 1)
            if self.msg_size > 0:
                x_red = torch.hstack((x, x_masked))
            else:
                x_red = x_masked
        else:
            x_red = x

        # Final output computation #
        output = self.mod_2.forward(x_red)
        return output

    def compute_compression_set(self, x):
        """
        Targets the examples that have the most contributed in the compression set.
        Args:
            x (torch.tensor of floats): input.
        """
        # Mask computation #
        if self.compression_set_size > 0:
            mask = self.cas[0].forward(x.clone())
            for j in range(1, len(self.cas)):
                out = self.cas[j].forward(x.clone())
                mask = torch.hstack((mask, out))
            self.msk = torch.squeeze(torch.topk(mask, 1, dim=2).indices)
        else:
            assert False, "Cannot compute the compression set when it is of size 0."
