import torch
from torch import nn as nn

from model.custom_attention import CA
from model.data_compressor import create_data_compressor_1
from model.kme import KME
from utils import MLP


class SimpleMetaNet(nn.Module):
    def __init__(self, pred_input_dim, config: dict) -> None:
        """
        Generates the DeepRM meta-predictor.
        Args:
            pred_input_dim (int): Input dimension of the predictor;
            config (dictionary) containing the following:
                m (int): Number of examples per dataset;
                d (int): Input dimension of each dataset;
                comp_set_size (int): compression set size;
                ca_dim (list of int): custom attention's MLP architecture;
                mod_1_dim (list of int): MLP #1 architecture;
                mod_2_dim (list of int): MLP #2 architecture;
                tau (int): temperature parameter (softmax in custom attention);
                batch_size (int): Batch size.
        """
        super(SimpleMetaNet, self).__init__()
        self.device = config["device"]
        self.comp_set_size = config["comp_set_size"]
        self.msg_size = config["msg_size"]
        self.ca_dim = config["ca_dim"]
        self.msg_size = config["msg_size"]
        self.mod_1_dim = config["mod_1_dim"] + [self.msg_size]
        self.mod_2_dim = config["mod_2_dim"]
        self.m = int(config["m"] / 2)
        self.msg_type = config["msg_type"]
        self.init_scheme = config["init_scheme"]
        self.msg = torch.tensor(0.0)  # Message (compression selection)
        self.msk = None  # Mask (compression selection)
        self.d = config["d"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.input_dim = self.d
        self.output_dim = pred_input_dim
        self.mod_2_input = config["data_compressor_dim"][-1] * (self.comp_set_size > 0) + self.mod_1_dim[-1] * (
                self.msg_size > 0)

        # Generating the many components (custom attention (CA) multi-heads, KME #1-2, MLP #1-2) of the meta-learner
        self.kme_1 = create_data_compressor_1(config)
        self.mod_1 = MLP(config["data_compressor_dim"][-1], self.mod_1_dim, self.device, config["has_skip_connection"],
                         config["has_batch_norm"], self.msg_type, self.init_scheme)

        self.cas = nn.ModuleList([])
        for i in range(self.comp_set_size):
            self.cas.append(CA(self.d + 1, self.ca_dim, self.ca_dim, self.m, self.device,
                               self.init_scheme, config["has_skip_connection"], config["has_batch_norm"], "fspool",
                               self.tau))
        self.kme_2 = KME(self.d + 1, config["data_compressor_dim"], self.device, self.init_scheme,
                         config["has_skip_connection"], config["has_batch_norm"])

        self.mod_2 = MLP(self.mod_2_input, self.mod_2_dim + [self.output_dim], self.device,
                         config["has_skip_connection"], config["has_batch_norm"], "none", self.init_scheme)

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

            # Passing through KME #1 #
            x = self.kme_1.forward(x)
            # Passing through MLP #1 #
            x = self.mod_1.forward(torch.reshape(x, (len(x), -1)))

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
        if self.comp_set_size > 0:
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
        if self.comp_set_size > 0:
            mask = self.cas[0].forward(x.clone())
            for j in range(1, len(self.cas)):
                out = self.cas[j].forward(x.clone())
                mask = torch.hstack((mask, out))
            self.msk = torch.squeeze(torch.topk(mask, 1, dim=2).indices)
        else:
            assert False, "Cannot compute the compression set when it is of size 0."
