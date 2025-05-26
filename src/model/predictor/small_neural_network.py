import torch
import torch.nn.functional as F
import torch.nn.parameter as parameter
from torch import nn
from torch.autograd import Variable

from src.model.lazy_batch_norm import LazyBatchNorm
from src.model.mlp import MLP
from src.model.predictor.predictor import Predictor


class FCNet(Predictor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.architecture_sizes = [config["n_features"]] + config["pred_hidden_sizes"] + [config["target_size"]]
        self.mlp = MLP(config["n_features"], self.architecture_sizes[1:], config["device"],
                       config["has_skip_connection"], config["has_batch_norm"], config["batch_norm_min_dim"])

        self.params = torch.tensor([])
        self.batch_norm_params = []
        self.prior = 0
        self.posterior_handicap = config["posterior_handicap"] if config["compute_prior"] else 1
        self.use_last_values = False
        self.save_bn_params = False
        self.target_size = config["target_size"]

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_params(self, params: torch.Tensor) -> None:
        self.params = params * self.posterior_handicap + self.prior

    def set_forward_mode(self, use_last_values: bool = False, save_bn_params: bool = False):
        self.use_last_values = use_last_values
        self.save_bn_params = save_bn_params

    def reset_forward_mode(self):
        self.use_last_values = False
        self.save_bn_params = False

    def get_batch_norm_from_prior(self, prior):
        for i in range(len(prior.mlp)):
            if isinstance(prior.mlp[i], LazyBatchNorm):
                self.mlp.mlp[i] = prior.mlp[i]

    def set_prior_weights(self, prior):
        first, weights = True, None
        for layer in prior.mlp:
            if isinstance(layer, nn.Linear):
                if first:
                    weights = layer.weight.data.clone().reshape((-1))
                    first = False
                else:
                    weights = torch.hstack((weights, layer.weight.data.clone().reshape((-1))))
                weights = torch.hstack((weights, layer.bias.data.clone().reshape((-1))))
        self.prior = weights.reshape((1, -1))

    def forward(self, instances: torch.Tensor) -> tuple:
        x = instances[:, :, :-self.target_size]
        meta_batch_size = len(x)

        if not self.use_last_values:
            self.batch_norm_params = []

        params_low_idx = 0
        params_high_idx = 0
        linear_layer_idx = 0
        for layer in self.mlp.get_modules():
            if isinstance(layer, nn.Linear):
                current_linear_layer_dim = self.architecture_sizes[linear_layer_idx]
                new_linear_layer_dim = self.architecture_sizes[linear_layer_idx + 1]

                params_high_idx += current_linear_layer_dim * new_linear_layer_dim
                weights = self.params[:, params_low_idx: params_high_idx].reshape(meta_batch_size,
                                                                                  new_linear_layer_dim,
                                                                                  current_linear_layer_dim)

                params_low_idx += current_linear_layer_dim * new_linear_layer_dim
                params_high_idx += new_linear_layer_dim
                bias = self.params[:, params_low_idx: params_high_idx].unsqueeze(dim=-2)

                weights = weights.transpose(-2, -1)
                x = x.matmul(weights) + bias

                params_low_idx += new_linear_layer_dim
                linear_layer_idx += 1

            elif isinstance(layer, LazyBatchNorm):
                if self.use_last_values:
                    x = layer.forward(x, is_using_saved_stats=self.use_last_values)
                elif self.save_bn_params:
                    x = layer.forward(x, is_saving_computed_stats=self.save_bn_params)
                else:
                    x = layer.forward(x)

            else:
                x = layer(x)
        return self._process_output(x)


def get_size_of_conv_output(input_shape, conv_func):
    # generate dummy input sample and forward to get shape after conv layers
    meta_batch_size = 1
    input = Variable(torch.rand(meta_batch_size, *input_shape))
    output_feat = conv_func(input)
    conv_out_size = output_feat.data.view(meta_batch_size, -1).size(1)
    return conv_out_size


class ConvNet(Predictor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.n_filt1 = int(config["pred_filter_sizes"][0])
        self.n_filt2 = int(config["pred_filter_sizes"][1])

        if "input_shape" not in config:
            error_msg = ("The class ConvNet is now unsupported. "
                         "You should add 'input_shape' in the .yaml config to make it work. "
                         "You will probably need to apply other fixes.")
            raise Exception(error_msg)

        self.input_shape = config["input_shape"]
        self.color_channels = config["input_shape"][0]

        if config["dataset"] == 'mnist_binary':  # TODO: support other dataset names
            conv_feat_size = 320
        elif config["dataset"] == 'cifar100':
            conv_feat_size = 500

        self.architecture_sizes = [conv_feat_size] + config["pred_hidden_sizes"] + [config["target_size"]]

        self.conv1 = nn.Conv2d(self.color_channels, self.n_filt1, kernel_size=5)
        self.conv2 = nn.Conv2d(self.n_filt1, self.n_filt2, kernel_size=5)
        self.mlp = MLP(conv_feat_size, self.architecture_sizes[1:], config["device"],
                       config["has_skip_connection"], config["has_batch_norm"], config["batch_norm_min_dim"])

        self.num_conv1_weight_params = torch.numel(self.conv1.weight)
        self.num_conv1_bias_params = torch.numel(self.conv1.bias)
        self.num_conv2_weight_params = torch.numel(self.conv2.weight)
        self.num_conv2_bias_params = torch.numel(self.conv2.bias)

        self.params = torch.tensor([])
        self.batch_norm_params = []
        self.prior = 0
        self.posterior_handicap = config["posterior_handicap"] if config["compute_prior"] else 1
        self.use_last_values = False
        self.save_bn_params = False
        self.target_size = config["target_size"]

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_params(self, params: torch.Tensor) -> None:
        self.params = params * self.posterior_handicap + self.prior

    def set_forward_mode(self, use_last_values: bool = False, save_bn_params: bool = False):
        self.use_last_values = use_last_values
        self.save_bn_params = save_bn_params

    def reset_forward_mode(self):
        self.use_last_values = False
        self.save_bn_params = False

    def get_batch_norm_from_prior(self, prior):
        for i in range(len(prior.mlp)):
            if isinstance(prior.mlp[i], LazyBatchNorm):
                self.mlp.mlp[i] = prior.mlp[i]

    def set_prior_weights(self, prior):
        first, weights = True, None
        for layer in prior.mlp:
            if isinstance(layer, nn.Linear):
                if first:
                    weights = layer.weight.data.clone().reshape((-1))
                    first = False
                else:
                    weights = torch.hstack((weights, layer.weight.data.clone().reshape((-1))))
                weights = torch.hstack((weights, layer.bias.data.clone().reshape((-1))))
        self.prior = weights.reshape((1, -1))

    def forward(self, instances: torch.Tensor) -> tuple:
        x = instances[:, :, :-self.target_size]
        x = x.reshape([x.shape[0], x.shape[1]] + self.input_shape)
        meta_batch_size = len(x)
        if not self.use_last_values:
            self.batch_norm_params = []

        rslts = []
        for dataset_idx in range(len(x)):
            self.conv1.weight = parameter.Parameter(self.params[dataset_idx, :self.num_conv1_weight_params].reshape(
                (self.n_filt1, self.color_channels, 5, 5)))
            self.conv1.bias = parameter.Parameter(self.params[dataset_idx, self.num_conv1_weight_params:
                                                                           self.num_conv1_weight_params + self.num_conv1_bias_params])
            self.conv2.weight = parameter.Parameter(
                self.params[dataset_idx, self.num_conv1_weight_params + self.num_conv1_bias_params:
                                         self.num_conv1_weight_params + self.num_conv1_bias_params + self.num_conv2_weight_params].reshape(
                    (self.n_filt2, self.n_filt1, 5, 5)))
            self.conv2.bias = parameter.Parameter(self.params[dataset_idx,
                                                  self.num_conv1_weight_params + self.num_conv1_bias_params + self.num_conv2_weight_params:
                                                  self.num_conv1_weight_params + self.num_conv1_bias_params + self.num_conv2_weight_params + self.num_conv2_bias_params])
            x_processed = F.elu(F.max_pool2d(self.conv1(x[dataset_idx]), 2))
            x_processed = F.elu(F.max_pool2d(self.conv2(x_processed), 2))
            x_processed = x_processed.view(x_processed.size(0), -1)
            rslts.append(x_processed)
        x = torch.stack(rslts)
        params_low_idx = self.num_conv1_weight_params + self.num_conv1_bias_params + self.num_conv2_weight_params + self.num_conv2_bias_params
        params_high_idx = self.num_conv1_weight_params + self.num_conv1_bias_params + self.num_conv2_weight_params + self.num_conv2_bias_params
        linear_layer_idx = 0
        for layer in self.mlp.get_modules():
            if isinstance(layer, nn.Linear):
                current_linear_layer_dim = self.architecture_sizes[linear_layer_idx]
                new_linear_layer_dim = self.architecture_sizes[linear_layer_idx + 1]
                params_high_idx += current_linear_layer_dim * new_linear_layer_dim
                weights = self.params[:, params_low_idx: params_high_idx].reshape(meta_batch_size,
                                                                                  new_linear_layer_dim,
                                                                                  current_linear_layer_dim)

                params_low_idx += current_linear_layer_dim * new_linear_layer_dim
                params_high_idx += new_linear_layer_dim
                bias = self.params[:, params_low_idx: params_high_idx].unsqueeze(dim=-2)

                weights = weights.transpose(-2, -1)
                x = x.matmul(weights) + bias

                params_low_idx += new_linear_layer_dim
                linear_layer_idx += 1

            elif isinstance(layer, LazyBatchNorm):
                if self.use_last_values:
                    x = layer.forward(x, is_using_saved_stats=self.use_last_values)
                elif self.save_bn_params:
                    x = layer.forward(x, is_saving_computed_stats=self.save_bn_params)
                else:
                    x = layer.forward(x)

            else:
                x = layer(x)
        return self._process_output(x)
