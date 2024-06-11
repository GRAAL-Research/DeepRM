from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


def lin_loss(output, targets, reduction=True):
    """
    Computes the linear loss.
    Args:
        output (torch.tensor of size (batch_size, m)): The output (0 or 1) of the predictor;
        targets (torch.tensor of size (batch_size, m)): The labels (0 or 1);
        reduction (bool): whether a mean should be applied
    Return:
        Float, the total linear loss incurred.
    """
    return torch.mean(((output * targets) + 1) / 2) if reduction == True else ((output * targets) + 1) / 2


def set_seed(seed):
    """
    Sets the seed to a certain value for several packages
    Args:
        seed (int): A seed
    """
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
            msg_type (str): type of message (choices: 'dsc' (discret), 'cnt' (continuous)).
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
            if k < len(self.dims) - 2:
                self.module.append(nn.ReLU())
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
        l = 0
        for layer in self.module:
            l += 1
            x_1 = x.clone()
            if self.skip and l == self.skip_position:
                x += x_1
            x = layer(x)
        return x


class SignStraightThrough(nn.Module):
    def __init__(self):
        """
        Implementation of the Straigh through estimator.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass for the Straigh through estimator.
        Args:
            input (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        out = input + torch.sign(input + 1e-20) - input.detach()
        out[torch.abs(input) > 1] = out[torch.abs(input) > 1].detach()
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
            out = torch.sum(torch.transpose(inputs[:, :, :-1], 0, 1)*self.weights[:, :-1], dim=-1) + self.weights[:, -1]
            out = torch.transpose(out, 0, 1)
        elif self.pred_type == 'small_nn':
            input = inputs[0, :, :-1]
            count_1, count_2, j = 0, 0, 0
            for layer in self.pred[0].module:
                if isinstance(layer, nn.Linear):
                    count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                    W = torch.reshape(self.weights[0, count_1:count_2], (self.pred_arch[j + 1], self.pred_arch[j]))
                    count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                    count_2 += self.pred_arch[j + 1]
                    b = torch.reshape(self.weights[0, count_1:count_2], (self.pred_arch[j + 1],))
                    count_1 += self.pred_arch[j + 1]
                    j += 1
                    input = torch.matmul(input, W.T) + b
                else:
                    input = layer(input)
            out = input
            for i in range(1, len(inputs)):
                input = inputs[i, :, :-1]
                count_1, count_2, j = 0, 0, 0
                for layer in self.pred[i].module:
                    if isinstance(layer, nn.Linear):
                        count_2 += self.pred_arch[j] * self.pred_arch[j + 1]
                        W = torch.reshape(self.weights[i, count_1:count_2], (self.pred_arch[j + 1], self.pred_arch[j]))
                        count_1 += self.pred_arch[j] * self.pred_arch[j + 1]
                        count_2 += self.pred_arch[j + 1]
                        b = torch.reshape(self.weights[i, count_1:count_2], (self.pred_arch[j + 1],))
                        count_1 += self.pred_arch[j + 1]
                        j += 1
                        input = torch.matmul(input, W.T) + b
                    else:
                        input = layer(input)
                out = torch.hstack((out, input))
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


def is_job_already_done(experiment_name, task_dict):
    """
    Verifies if a hyperparameter combination has already been tested.
    Args:
        experiment_name (str): The name of the experiment;
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
        new.append(str(dict[key]))
    try:
        with open("results/" + str(experiment_name) + ".txt", "r") as tes:
            tess = [line.strip().split('\t') for line in tes]
        tes.close()
        for i in range(len(tess)):
            if tess[i][:len(new)] == new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open("results/" + str(experiment_name) + ".txt", "a")
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
    max_number_vis = 16     # Maximum number of decision boundaries to compute
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        examples = []
        for inputs_1, targets_1 in data_loader:
            for j in range(len(inputs_1)):
                if i < max_number_vis:
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()
                    i += 1
                    inputs_1, targets_1 = inputs_1.float(), targets_1.float()
                    inds = inputs_1[j, :, -1].sort().indices.tolist()   # Sorts the examples by their labels
                    # ... so that each class can be plotted with different colours
                    X = inputs_1[j, inds][:, :2]
                    m = int(len(X) / 2)
                    if str(device) == 'gpu':
                        inputs_1, targets_1, meta_pred = inputs_1.cuda(), targets_1.cuda(), meta_pred.cuda()
                    meta_output = meta_pred(inputs_1[:, :m])[j]
                    if pred.pred_type == 'linear_classif':
                        px = [-20, 20]
                        py = [-(-20 * meta_output[0] + meta_output[2]) / meta_output[1],
                              -(20 * meta_output[0] + meta_output[2]) / meta_output[1]]
                        plt.plot(px, py)    # With a linea classifier, only a line needs to be draw
                    if pred.pred_type == 'small_nn':
                        # With small nn: we plot the decision boundary by colouring each decision zone by its prediction
                        h = .05  # step size in the mesh
                        x_min, x_max = X[:, 0].cpu().min() - 10, X[:, 0].cpu().max() + 10
                        y_min, y_max = X[:, 1].cpu().min() - 10, X[:, 1].cpu().max() + 10
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                             np.arange(y_min, y_max, h))
                        input = np.array(np.c_[xx.ravel(), yy.ravel()])
                        input = np.hstack((input, np.ones((len(input), 1))))
                        input = torch.from_numpy(np.array([input]))
                        if str(device) == 'gpu':
                            input = input.cuda()
                        Z = pred.forward(input.double(), torch.reshape(meta_output, (1, -1)).double())
                        Z = torch.round(Z.reshape(xx.shape)).cpu()
                        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
                    plt.scatter(X[m:, 0].cpu(), X[m:, 1].cpu(), c='r')
                    plt.scatter(X[:m, 0].cpu(), X[:m, 1].cpu(), c='b')
                    if meta_pred.k > 0:
                        meta_pred.compute_compression_set(inputs_1[:, :m])
                        plt.scatter(X[meta_pred.msk[j].cpu(), 0].cpu(),
                                    X[meta_pred.msk[j].cpu(), 1].cpu(), c='black', s=120, marker='*')
                    if dataset in ['easy', 'hard']:
                        plt.xlim(-20, 20)
                        plt.ylim(-20, 20)
                    if dataset == 'moons':
                        plt.xlim(torch.mean(X[:, 0].cpu()) - 10, torch.mean(X[:, 0].cpu()) + 10)
                        plt.ylim(torch.mean(X[:, 1].cpu()) - 10, torch.mean(X[:, 1].cpu()) + 10)
                    plt.savefig(f"figures/decision_boundaries/decision_boundaries_{i}.png")
                    if wandb is not None:
                        im_frame = Image.open(f"figures/decision_boundaries/decision_boundaries_{i}.png")
                        image = wandb.Image(np.array(im_frame),
                                            caption=f"decision_boundaries/decision_boundaries_{i}")  # file_type="jpg"
                        examples.append(image)
    wandb.log({"Decision boundaries": examples})
