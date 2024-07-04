import torch
from torch import nn as nn

from src.model.mlp import MLP


class Predictor(nn.Module):
    def __init__(self, config: dict):
        """
        Generates the predictor, a feed-forward ReLU network (linear classifier, if the number of hidden layers is 0).
        """
        super(Predictor, self).__init__()
        self.d = config["d"]
        self.batch_size = config["batch_size"]
        self.has_skip_connection = config["has_skip_connection"]
        self.has_batch_norm = config["has_batch_norm"]

        pred_arch = config["pred_arch"]
        self.pred_type = "linear_classif" if len(pred_arch) == 0 else "small_nn"
        self.weights = []
        # It is useful to know how many parameters there is; the architecture now contains the input dim and output dim
        self.num_param, self.pred_arch = self.num_param_arch_init(self.d, pred_arch)
        self.pred = self.pred_init(self.batch_size)

    def num_param_arch_init(self, d, pred_arch):
        """
        Computes the total number of parameters defining the predictor, and the architecture (with input and output dim)
        Args:
            d (int): Input dimension of each dataset;
            pred_arch (list of int): architecture of the predictor.
        Return:
            int, number of parameters defining the predictor;
            list, architecture of the predictor.
        """
        num_param, arch = 0, []
        if self.pred_type == "linear_classif":
            num_param = d + 1
        if self.pred_type == "small_nn":
            arch = [self.d] + pred_arch + [1]
            for i in range(1, len(arch)):
                num_param += (arch[i - 1] + 1) * arch[i]
        return num_param, arch

    def pred_init(self, batch_size):
        """
        Initialize the predictors: one predictor per dataset in a batch. Only relevant if pred_type == "small_nn".
        Args:
            batch_size (int): Batch size.
        Return:
            list of torch.nn.ModuleList, one predictor per dataset in a batch.
        """
        structure = []
        if self.pred_type == "small_nn":
            for i in range(batch_size):
                mlp = MLP(self.d, self.pred_arch[1:], "cpu", self.has_skip_connection, self.has_batch_norm, "none")
                structure.append(mlp)

        return structure

    def update_weights(self, weights, batch_size):
        """
        Fixes the weights of the various predictors.
        Args:
            weights (np.array of dims (batch_size, num_param) of float): weights defining the predictor;
            batch_size (int): batch size.
        """
        self.weights = weights
        if self.pred_type == "small_nn":
            for i in range(batch_size):
                count_1, count_2, j = 0, 0, 0
                for layer in self.pred[i].module:
                    if isinstance(layer, nn.Linear):
                        count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                        layer.weight.data = torch.reshape(self.weights[i, count_1:count_2],
                                                          (self.pred_arch[j + 1], self.pred_arch[j]))
                        count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                        count_2 += self.pred_arch[j + 1]
                        layer.bias.data = torch.reshape(self.weights[i, count_1:count_2], (self.pred_arch[j + 1],))
                        count_1 += self.pred_arch[j + 1]
                        j += 1

    def forward(self, inputs, return_sign=False):
        """
        Computes a forward pass of the various predictor (one per dataset in a given batch).
        Args:
            inputs (): ;
            return_sign (bool): whether to round the predictions or not.
        Return:
            torch.Tensor of dims (batch_size, m, output_dims), the predictions.
        """
        out = 0
        if self.pred_type == "linear_classif":
            out = torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1) * self.weights[:, :-1], dim=-1) + self.weights[:,
                                                                                                       -1]
            out = torch.transpose(out, 0, 1)
        elif self.pred_type == "small_nn":
            input_0 = inputs[0, :, :-1]
            count_1, count_2, j = 0, 0, 0
            for layer in self.pred[0].module:
                if isinstance(layer, nn.Linear):
                    count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                    w = torch.reshape(self.weights[0, count_1:count_2], (self.pred_arch[j + 1], self.pred_arch[j]))
                    count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                    count_2 += self.pred_arch[j + 1]
                    b = torch.reshape(self.weights[0, count_1:count_2], (self.pred_arch[j + 1],))
                    count_1 += self.pred_arch[j + 1]
                    j += 1
                    input_0 = torch.matmul(input_0, w.T) + b
                else:
                    input_0 = layer(input_0)
            out = input_0
            for i in range(1, len(inputs)):
                input_i = inputs[i, :, :-1]
                count_1, count_2, j = 0, 0, 0
                for layer in self.pred[i].module:
                    if isinstance(layer, nn.Linear):
                        count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                        w = torch.reshape(self.weights[i, count_1:count_2], (self.pred_arch[j + 1], self.pred_arch[j]))
                        count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                        count_2 += self.pred_arch[j + 1]
                        b = torch.reshape(self.weights[i, count_1:count_2], (self.pred_arch[j + 1],))
                        count_1 += self.pred_arch[j + 1]
                        j += 1
                        input_i = torch.matmul(input_i, w.T) + b
                    else:
                        input_i = layer(input_i)
                out = torch.hstack((out, input_i))
            out = torch.transpose(out, 0, 1)
        if not return_sign:
            return torch.sigmoid(out)
        return torch.sigmoid(out), torch.sign(out)
