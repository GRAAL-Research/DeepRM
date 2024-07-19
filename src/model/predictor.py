import torch
from torch import nn as nn

from src.model.mlp import MLP
from src.model.lazy_batch_norm import LazyBatchNorm


class Predictor(nn.Module):
    def __init__(self, config: dict):
        """
        Generates the predictor, a feed-forward ReLU network (linear classifier, if the number of hidden layers is 0).
        """
        super(Predictor, self).__init__()
        self.pred_type = "linear_classif" if len(config["pred_hidden_sizes"]) == 0 else "small_nn"
        self.weights = []
        # It is useful to know how many parameters there is; the architecture now contains the input dim and output dim
        self.n_param, self.pred_arch = self.compute_n_params_and_arch_sizes(config["n_features"],
                                                                            config["pred_hidden_sizes"])
        self.pred = self.create_predictor(config)
        self.task = config["task"]

    def compute_n_params_and_arch_sizes(self, n_features: int, pred_hidden_sizes: list[int]) -> tuple[int, list[int]]:
        """
        Computes the total number of parameters defining the predictor, and complete arch (with input and output dim)

        n_features (int): Input dimension of each dataset;
        """
        if self.pred_type == "linear_classif":
            n_params = n_features + 1
            architecture_sizes = []
            return n_params, architecture_sizes

        elif self.pred_type == "small_nn":
            n_params = 0
            input_layer_size = n_features
            output_layer_size = 1
            architecture_sizes = [input_layer_size] + pred_hidden_sizes + [output_layer_size]

            for i in range(len(architecture_sizes) - 1):
                nb_of_bias = 1
                current_layer_nb_of_neurons_and_bias = architecture_sizes[i] + nb_of_bias
                n_params += current_layer_nb_of_neurons_and_bias * architecture_sizes[i + 1]

            return n_params, architecture_sizes

        raise NotImplementedError(f"The predictor type '{self.pred_type}' is not supported.")

    def create_predictor(self, config: dict) -> MLP:
        """
        Initialize the predictors: one predictor per dataset in a batch. Only relevant if pred_type == "small_nn".
        """
        if self.pred_type == "small_nn":
            mlp = MLP(config["n_features"], self.pred_arch[1:], config["device"], config["has_skip_connection"],
                      config["has_batch_norm"], "none")
        return mlp


    def set_weights(self, weights):
        """
        Fixes the weights of the various predictors.
        Args:
            weights (np.array of dims (batch_size, num_param) of float): weights defining the predictor;
            batch_size (int): batch size.
        """
        self.weights = weights

    def forward(self, inputs, use_last_values=False, save_bn_params=False):
        if not use_last_values:
            self.batch_norm_params = []
        out = 0
        if self.pred_type == "linear_classif":
            out = (torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1) * self.weights[:, :-1], dim=-1)
                   + self.weights[:, -1])
            out = torch.transpose(out, 0, 1)
        elif self.pred_type == "small_nn":
            batch_size = len(inputs)
            input_0 = inputs[:, :, :-1]
            count_1, count_2, n_batch_norm, j = 0, 0, 0, 0
            for layer in self.pred.module:
                if isinstance(layer, nn.Linear):
                    count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                    w = torch.reshape(self.weights[:, count_1:count_2],
                                      (batch_size, self.pred_arch[j + 1], self.pred_arch[j]))
                    count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                    count_2 += self.pred_arch[j + 1]
                    b = torch.reshape(self.weights[:, count_1:count_2], (batch_size, 1, self.pred_arch[j + 1]))
                    count_1 += self.pred_arch[j + 1]
                    j += 1
                    w = torch.transpose(w, 1, 2)
                    input_0 = torch.matmul(input_0, w) + b
                elif isinstance(layer, LazyBatchNorm):
                    if use_last_values:
                        input_0 = layer.forward(input_0, use_last_values=use_last_values)
                    elif save_bn_params:
                        input_0 = layer.forward(input_0, save_bn_params=save_bn_params)
                    else:
                        input_0 = layer.forward(input_0)
                    n_batch_norm += 1
                else:
                    input_0 = layer(input_0)
            out = input_0
            out = torch.squeeze(torch.transpose(out, 1, 2))
        if self.task == "classification":
            return torch.sigmoid(out), torch.sign(out)
        elif self.task == "regression":
            return torch.relu(out), out
