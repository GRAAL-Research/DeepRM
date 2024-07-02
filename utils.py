import random
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

plt.switch_backend('agg')
import torch
import torch.nn as nn


def lin_loss(output, targets):
    """
    Computes the linear loss.
    Args:
        output (torch.tensor of size (batch_size, m)): The output (0 or 1) of the predictor;
        targets (torch.tensor of size (batch_size, m)): The labels (0 or 1);
    Return:
        Float, the total linear loss incurred.
    """
    tot = 0
    if torch.sum(targets == 1) > 0:
        tot += torch.mean(((output[targets == 1] * targets[targets == 1]) + 1) / 4, dim=0)
    else:
        tot += torch.tensor(0.5)
        print("This batch contains only examples from a single class.")
    if torch.sum(targets == -1) > 0:
        tot += torch.mean(((output[targets == -1] * targets[targets == -1]) + 1) / 4, dim=0)
    else:
        tot += torch.tensor(0.5)
        print("This batch contains only examples from a single class.")
    return tot


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def l1(x, c):
    """
    Computes the l1 loss, given inputs and a regularization parameter.
    Args:
        x (torch.tensor of size m): Inputs
        c (float): Regularization parameter
    Return:
        Float, the l1 loss
    """
    return torch.mean(torch.abs(x)) * c


def l2(x, c):
    """
    Computes the l1 loss, given inputs and a regularization parameter.
    Args:
        x (torch.tensor of size m): Inputs
        c (float): Regularization parameter
    Return:
        Float, the l2 loss
    """
    return torch.mean(x ** 2) * c


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, device, init, skip, bn, msg_type):
        """
        Creates a ReLU linear layer, given dimensions.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the MLP;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not;
            msg_type (str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous)).
        return:
            torch.nn.Module
        """
        super(MLP, self).__init__()
        self.dims = [input_dim] + hidden_dims
        self.skip = skip
        self.module = torch.nn.ModuleList()
        if self.skip and len(self.dims) > 2:
            self.dims.insert(len(self.dims) - 1, self.dims[0])
        for k in range(len(self.dims) - 1):
            if bn:
                self.module.append(nn.LazyBatchNorm1d())
            self.module.append(nn.Linear(self.dims[k], self.dims[k + 1]))
            if k < len(self.dims) - 2 or msg_type == 'pos':
                self.module.append(nn.ReLU())
            elif k == len(self.dims) - 2 and msg_type == 'none':
                self.module.append(nn.Identity())
            elif k == len(self.dims) - 2 and msg_type == 'dsc':
                self.module.append(SignStraightThrough())
            elif k == len(self.dims) - 2 and msg_type == 'cnt':
                self.module.append(nn.Tanh())

        self.skip_position = len(self.module) - (1 + 1 * bn)
        if device == 'gpu':
            self.module.to('cuda:0')
        if init is not None:
            init_weights(init, self.module)

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        lay_cnt = 0
        for layer in self.module:
            lay_cnt += 1
            x_1 = x.clone()
            if self.skip and lay_cnt == self.skip_position:
                x += x_1
            x = layer(x).clone()
        return x


class SignStraightThrough(nn.Module):
    def __init__(self):
        """
        Implementation of the Straight through estimator.
        """
        super().__init__()

    @staticmethod
    def forward(inputs):
        """
        Forward pass for the Straight through estimator.
        Args:
            inputs (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        out = torch.sign(inputs + 1e-20) + inputs - inputs.detach()
        out[torch.abs(inputs) > 1] = out[torch.abs(inputs) > 1].detach()
        return out


def init_weights(init, module):
    """
    Initialize the weights of the linear layers of one or many torch.nn.Module.
    Args:
        init (str): Layer initialization scheme (choices: 'kaiming_unif', '.._norm', 'xavier_unif', '.._norm')
        module (torch.nn.Module): The module of interest
    """
    for layer in module:
        if isinstance(layer, nn.Linear):
            if init == 'kaiming_unif':
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
            elif init == 'kaiming_norm':
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
            elif init == 'xavier_unif':
                nn.init.xavier_uniform_(layer.weight.data)
            elif init == 'xavier_norm':
                nn.init.xavier_normal_(layer.weight.data)


class Predictor(nn.Module):
    def __init__(self, d, pred_arch, batch_size):
        """
        Generates the predictor, a feed-forward ReLU network (linear classifier, if the number of hidden layers is 0).
        Args:
            d (int): Input dimension of each dataset;
            pred_arch (list of int): architecture of the predictor;
            batch_size (int): Batch size.
        """
        super(Predictor, self).__init__()
        self.pred_type = 'linear_classif' if len(pred_arch) == 0 else 'small_nn'
        self.d, self.batch_size, self.weights = d, batch_size, []
        # It is useful to know how many parameters there is; the architecture now contains the input dim and output dim
        self.num_param, self.pred_arch = self.num_param_arch_init(d, pred_arch)
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
        if self.pred_type == 'linear_classif':
            num_param = d + 1
        if self.pred_type == 'small_nn':
            arch = [self.d] + pred_arch + [1]
            for i in range(1, len(arch)):
                num_param += (arch[i - 1] + 1) * arch[i]
        return num_param, arch

    def pred_init(self, batch_size):
        """
        Initialize the predictors: one predictor per dataset in a batch. Only relevant if pred_type == 'small_nn'.
        Args:
            batch_size (int): Batch size.
        Return:
            list of torch.nn.ModuleList, one predictor per dataset in a batch.
        """
        structure = []
        if self.pred_type == 'small_nn':
            for i in range(batch_size):
                structure.append(MLP(self.d, self.pred_arch[1:], 'cpu', 'None', False, False, 'None'))
        return structure

    def update_weights(self, weights, batch_size):
        """
        Fixes the weights of the various predictors.
        Args:
            weights (np.array of dims (batch_size, num_param) of float): weights defining the predictor;
            batch_size (int): batch size.
        """
        self.weights = weights
        if self.pred_type == 'small_nn':
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
        if self.pred_type == 'linear_classif':
            out = torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1) * self.weights[:, :-1], dim=-1) + self.weights[:,
                                                                                                       -1]
            out = torch.transpose(out, 0, 1)
        elif self.pred_type == 'small_nn':
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


def update_hist(hist, values):
    """
    Adds values to the hist. dictionary to keep track of the losses and accuracies for each epoch.
    Args:
        hist (dic): A dictionary that keep track of training metrics.
        values (Tuple): Elements to be added to the dictionary.
    """
    hist['train_acc'].append(values[0])
    hist['train_loss'].append(values[1])
    hist['valid_acc'].append(values[2])
    hist['valid_loss'].append(values[3])
    hist['test_acc'].append(values[4])
    hist['test_loss'].append(values[5])
    hist['bound_lin'].append(values[6][0])
    hist['bound_hyp'].append(values[6][1])
    hist['bound_kl'].append(values[6][2])
    hist['bound_mrch'].append(values[6][3])


def update_wandb(wandb, hist):
    """
    Upload values to WandB.
    Args:
        wandb (package): the weights and biases package;
        hist (dic): A dictionary that keep track of training metrics.
    """
    wandb.log({'train_acc': hist['train_acc'][-1],
               'train_loss': hist['train_loss'][-1],
               'valid_acc': hist['valid_acc'][-1],
               'valid_loss': hist['valid_loss'][-1],
               'test_acc': hist['test_acc'][-1],
               'test_loss': hist['test_loss'][-1],
               'bound_lin': hist['bound_lin'][-1],
               'bound_hyp': hist['bound_hyp'][-1],
               'bound_kl': hist['bound_kl'][-1],
               'bound_mrch': hist['bound_mrch'][-1]})


def write(file_name, task_dict, hist, best_epoch):
    """
    Writes in a .txt file the hyperparameters and results of a training of the BGN algorithm
        on a given dataset.
    Args:
        file_name (str): The name of the .txt file to write into;
        task_dict (dictionary): the dictionary containing the current hyperparameters combination;
        hist (dictionary): A dictionary that keep track of training metrics.
        best_epoch (int): best epoch.
    """
    keys = []
    for key in task_dict:
        keys.append(key)
    keys.sort()
    file = open("results/" + str(file_name) + ".txt", "a")
    for key in keys:
        file.write(str(task_dict[key]) + "\t")
    file.write(str(hist['train_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['valid_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['test_acc'][best_epoch].item()) + "\t")
    file.write(str(hist['bound_lin'][best_epoch].item()) + "\t")
    file.write(str(hist['bound_hyp'][best_epoch].item()) + "\t")
    file.write(str(hist['bound_kl'][best_epoch]) + "\t")
    file.write(str(hist['bound_mrch'][best_epoch]) + "\t")
    file.write("\n")
    file.close()


def is_job_already_done(project_name, task_dict):
    """
    Verifies if a hyperparameter combination has already been tested.
    Args:
        project_name (str): The name of the wandb project;
        task_dict (dictionary): the dictionary containing the current hyperparameters combination;
    Return:
        bool, whether the combination has already been tested or not
    """
    cnt_nw = 0
    new, keys = [], []
    for key in task_dict:
        keys.append(key)
    keys.sort()
    for key in keys:
        new.append(str(task_dict[key]))
    try:
        with open("results/" + str(project_name) + ".txt", "r") as tes:
            tess = [line.strip().split('\t') for line in tes]
        tes.close()
        for i in range(len(tess)):
            if tess[i][:len(new)] == new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open("results/" + str(project_name) + ".txt", "a")
        for key in keys:
            file.write(key + "\t")
        file.write('train_acc' + "\t")
        file.write('valid_acc' + "\t")
        file.write('test_acc' + "\t")
        file.write('bound_lin' + "\t")
        file.write('bound_hyp' + "\t")
        file.write('bound_kl' + "\t")
        file.write('bound_mrch' + "\t")
        file.write('n_sigma' + "\t")
        file.write('n_Z' + "\n")
        file.close()
    return cnt_nw > 0


def show_decision_boundaries(meta_pred, dataset, data_loader, pred, wandb, device):
    """
    Builds a visual depiction of the decision boundary of the predictor for tackling a given problem.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train;
        dataset (str): name of the current dataset;
        data_loader (DataLoader): A DataLoader to test on;
        pred (Predictor): the predictor;
        wandb (package): the weights and biases package;
        device (str): 'gpu', or 'cpu'; whether to use the gpu.
    """
    max_number_vis = 16  # Maximum number of decision boundaries to compute
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        examples = []
        for inputs in data_loader:
            for j in range(len(inputs)):
                if i < max_number_vis:
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()
                    i += 1
                    targets = (inputs.clone()[:, :, -1] + 1) / 2
                    inputs, targets = inputs.float(), targets.float()
                    inds = inputs[j, :, -1].sort().indices.tolist()  # Sorts the examples by their labels
                    # ... so that each class can be plotted with different colours
                    x = inputs[j, inds][:, :2]
                    m = int(len(x) / 2)
                    if str(device) == 'gpu':
                        inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                    meta_output = meta_pred(inputs[:, :m])[j:j + 1]
                    if pred.pred_type == 'linear_classif':
                        px = [-20, 20]
                        py = [-(-20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1],
                              -(20 * meta_output[0, 0] + meta_output[0, 2]) / meta_output[0, 1]]
                        plt.plot(px, py)  # With a linea classifier, only a line needs to be drawn
                    if pred.pred_type == 'small_nn':
                        # With small nn: we plot the decision boundary by colouring each decision zone by its prediction
                        h = .05  # step size in the mesh
                        x_min, x_max = x[:, 0].cpu().min() - 10, x[:, 0].cpu().max() + 10
                        y_min, y_max = x[:, 1].cpu().min() - 10, x[:, 1].cpu().max() + 10
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                             np.arange(y_min, y_max, h))
                        mesh = np.array(np.c_[xx.ravel(), yy.ravel()])
                        mesh = np.hstack((mesh, np.ones((len(mesh), 1))))
                        mesh = torch.from_numpy(np.array([mesh])).float()
                        if str(device) == 'gpu':
                            mesh = mesh.cuda()
                        pred.update_weights(meta_output, 1)
                        z = pred.forward(mesh)
                        z = torch.round(z.reshape(xx.shape)).cpu()
                        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.6)
                    plt.scatter(x[m:, 0].cpu(), x[m:, 1].cpu(), c='r')
                    plt.scatter(x[:m, 0].cpu(), x[:m, 1].cpu(), c='b')
                    if meta_pred.comp_set_size > 0:
                        meta_pred.compute_compression_set(inputs[:, :m])
                        plt.scatter(x[meta_pred.msk[j].cpu(), 0].cpu(),
                                    x[meta_pred.msk[j].cpu(), 1].cpu(), c='black', s=120, marker='*')
                    if dataset == "blob":
                        plt.xlim(-20, 20)
                        plt.ylim(-20, 20)
                    if dataset == 'moon':
                        plt.xlim(torch.mean(x[:, 0].cpu()) - 10, torch.mean(x[:, 0].cpu()) + 10)
                        plt.ylim(torch.mean(x[:, 1].cpu()) - 10, torch.mean(x[:, 1].cpu()) + 10)

                    figure_folder_path = Path("figures")
                    if not figure_folder_path.exists():
                        figure_folder_path.mkdir()

                    decision_boundaries_folder_name = "decision_boundaries"
                    decision_boundaries_folder_path = figure_folder_path / decision_boundaries_folder_name
                    if not decision_boundaries_folder_path.exists():
                        decision_boundaries_folder_path.mkdir()

                    fig_prefix = "decision_boundaries"
                    plt.savefig(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                    if wandb is not None:
                        im_frame = Image.open(decision_boundaries_folder_path / f"{fig_prefix}_{i}.png")
                        image = wandb.Image(np.array(im_frame),
                                            caption=f"{decision_boundaries_folder_name}/{fig_prefix}_{i}")  # file_type="jpg"
                        examples.append(image)

    wandb.log({"Decision boundaries": examples})
