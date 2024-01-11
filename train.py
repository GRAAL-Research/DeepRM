import math
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from copy import copy
from bound import compute_bound
import wandb


class SimpleMetaNet(nn.Module):
    def __init__(self, input_dim, dims, output_dim, m, d, k, tau, batch_size, init, device):
        """
        Generates a simple feed-forward neural network with ReLU activations and kernel mean embedding (MLP).
            The kernel has |kernel_dim| layer; the element outputted by the Kernel are multiplied by the label values
            before applying the mean function. Then, there are |hidden_dim| hidden layers, and an output layer.
        Args:
            intput_dim (int): Input dimension of the network.
            dims (tuple of list of int):
            output_dim (int): Outputput dimension of the network
            m (int): Number of examples per dataset
            d (int): Input dimension of each dataset
            k (int): Sample compression size
            tau (float): Temperature parameter for the Gumbel-Softmax component
            batch_size (int): Batch size.
            init (string): Layer initialization scheme (choices: 'kaiming_unif', '.._norm', 'xavier_unif', '.._norm')
        Output:
            nn.Module A feedforward neural network.
        """
        super(SimpleMetaNet, self).__init__()
        if device == "gpu" and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.msg, self.msk = None, None  # Message and mask (for compression selection)
        self.kern_1_dim, self.kern_2_dim, self.modl_1_dim, self.modl_2_dim = dims  # Dims of all 4 components
        self.m, self.d, self.k, self.tau, self.batch_size = m, d, k, tau, batch_size  # Parameters
        assert 0 <= k <= m  # Sample compression size is bounded
        assert self.kern_2_dim[-1] == self.modl_2_dim[-1]  # Since there is a skip connection, must be of same size
        self.output_dim_1 = self.modl_1_dim[-1] + self.m  # So that input size is (message size + mask size)
        self.output_dim_2 = output_dim
        self.modl_2_dim[-1] = self.kern_2_dim[-1] + self.modl_1_dim.copy()[-1]
        self.modl_1_dim[-1] = self.kern_1_dim[-1]

        self.kern_1 = relu_fc_layer([input_dim] + self.kern_1_dim, self.device)  # The different layers are instanciated
        self.kern_2 = relu_fc_layer([input_dim] + self.kern_2_dim, self.device)
        self.modl_1 = relu_fc_layer([self.kern_1_dim[-1]] + self.modl_1_dim, self.device)
        self.modl_2 = relu_fc_layer([self.output_dim_1 - self.m + self.kern_2_dim[-1]] + self.modl_2_dim, self.device,
                                    last=True)
        self.last_l_1 = relu_fc_layer([self.modl_1_dim[-1]] + [self.output_dim_1], self.device)
        self.last_l_2 = relu_fc_layer([self.modl_2_dim[-1]] + [self.output_dim_2], self.device)
        init_weights(init, [self.kern_1, self.kern_2, self.modl_1, self.modl_2])

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input
        return:
            torch.tensor Output of the network
        """
        ## First KME ##
        x_t = torch.zeros((self.batch_size, self.kern_1_dim[-1])).to(
            torch.device(self.device))  # Size of the KME output
        for i in range(len(x)):  # For each dataset ...
            x_i = x[i][:, :-1]
            for layer in self.kern_1:
                x_i = layer(x_i)
            x_t[i] = torch.mean(x_i * torch.reshape(x[i][:, -1], (-1, 1)), dim=0)  # ... A compression is done

        ## First module ##
        skip = x_t[:, :]  # A skip connection is made
        for layer in self.modl_1:
            x_t = layer(x_t)
        for layer in self.last_l_1:
            x_t = layer(x_t + skip)  # The skip connection leads to just before applying the last linear layer

        ## Compression set computation ##
        if int(self.k) == self.k:
            for i in range(len(x_t)):
                odds = F.gumbel_softmax(x_t[i, :self.m], tau=self.tau, hard=False)
                inds = torch.topk(odds, self.k).indices  # We retain the top-k examples having the highest odds
                tens = torch.zeros(len(odds)).to(torch.device(self.device))
                tens[[inds]] += 1
                x_t[i, :self.m] = tens - odds.detach() + odds  # Straight-trough estimator
        else:
            for i in range(len(x_t)):
                odds = F.gumbel_softmax(x_t[i, :self.m], tau=self.tau, hard=False)
                inds = (odds > self.k)  # We retain the examples having higher odds than k
                tens = torch.zeros(len(odds))
                tens[[inds]] += 1
                x_t[i, :self.m] = tens - odds.detach() + odds  # Straight-trough estimator
        self.msk = torch.clone(x_t[:, :self.m])
        self.msg = x_t[:, self.m:]

        ## Second KME ##
        x_t_2 = torch.zeros((self.batch_size, self.kern_2_dim[-1])).to(torch.device(self.device))
        for i in range(len(x)):  # For each dataset ...
            x_i = torch.zeros((self.m, self.d)).to(torch.device(self.device))
            for j in range(self.m):
                x_i[j] = x[i][j, :-1] * x_t[i, j]
            for layer in self.kern_2:
                x_i = layer(x_i)
            x_t_2[i] = torch.mean(x_i * torch.reshape(x[i][:, -1], (-1, 1)), dim=0)  # ... A compression is done
        x_t = torch.hstack((x_t_2, x_t[:, self.m:]))  # Both the compression set (after going through the 2nd KME) and
        #     the message are given to the second module as inputs.

        ## Second module ##
        skip = x_t[:, :]  # [:, :self.kern_2_dim[-1]]  # A skip connection is made
        for layer in self.modl_2:
            x_t = layer(x_t)
        for layer in self.last_l_2:
            x_t = layer(x_t + skip)  # The skip connection leads to just before applying the last linear layer
        return x_t

    def print_weights(self):
        for layer in self.kern_1:
            if isinstance(layer, nn.Linear):
                print(layer.weight.data[0, :5], layer.bias)
                break

    def print_grad(self):
        for layer in self.kern_1:
            if isinstance(layer, nn.Linear):
                print(layer.weight.grad[0, :5], layer.bias.grad[:5])
                break


def relu_fc_layer(dims, device, last=False):
    """
    Creates a ReLU linear layer, given dimensions.
    Args:
        dims (vector of ints): (respectively) size of the input layer, of the hidden layers, and of the output layer
        last (bool): Whether (True) or not (False) to put an activation function at the end of the last layer
    return:
        torch.nn.Module
    """
    lay = torch.nn.ModuleList()
    lay.to(torch.device(device))
    for i in range(len(dims) - 1):
        lay.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last:
            lay.append(nn.ReLU())
    return lay


def init_weights(init, modules):
    """
    Instanciate the weights of the linear layers of one or many torch.nn.Module.
    Args:
        init (str): Layer initialization scheme (choices: 'kaiming_unif', '.._norm', 'xavier_unif', '.._norm')
        modules (list of torch.nn.Module): The module(s) of interest
    """
    for module in modules:
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


def train_valid_loaders(dataset, batch_size, test_split=0.25, valid_split=0.2, shuffle=True, seed=42):
    """
    Divides a dataset into a training set and a validation set, both in a Pytorch DataLoader form.
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Desired batch-size for the DataLoader
        train_split (float): Desired proportion of training example.
        shuffle (bool): Whether the examples are shuffled before train/validation split.
        seed (int): A random seed.
    Returns:
        Tuple (training DataLoader, validation DataLoader).
    """
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split_1 = math.floor(test_split * num_data)
    split_2 = math.floor(test_split * valid_split * num_data)
    test_idx, valid_idx, train_idx = indices[:split_1], indices[split_1:split_1 + split_2], indices[split_1 + split_2:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def train(meta_pred, pred, data, dataset, test_split, valid_split, optimizer, scheduler, tol, early_stop, n_epoch,
          batch_size,
          criterion, penalty_msg, penalty_msg_coef, vis, vis_loss_acc, bound_type, DEVICE, independent_food,
          weightsbiases):
    """
    Trains a meta predictor using PyTorch.

    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (function): A predictor whose parameters are computed by the meta predictor.
        dataset (Dataset): A dataset.
        test_split (float): Desired proportion of test example.
        valid_split (float): Desired proportion of validation examples (in the remaining).
        optimizer (torch.optim): Optimisation algorithm
        scheduler (torch.optim.lr_scheduler): Scheduler for the learning rate
        tol (float): Quantity by which the loss must diminishes in order for this increment not to be marginal
        early_stop (int): Number of epochs by which, if the loss hasn't diminished by tol (see above), the training stops
        n_epoch (int): The maximum number of epochs.
        batch_size (int): Batch size.
        criterion (torch.nn, function): Loss function
        penalty_msg (func): A regularization function for the message. (See utils.l1 for an example of such function)
        penalty_msg_coef (float): Message regularization factor.
        vis (int): Number of created predictors to visualize (plot) at the end of the training (only works for
                        linear classifier with 2-d datasets).
        bound_type (str): The type of PAC bound to compute (choices: "Alex", "Mathieu").
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu

    Returns:
        Tuple (nn.Module, dict): The trained model; dictionary containing several information about the training
    """
    train_loader, valid_loader, test_loader = train_valid_loaders(data, batch_size, test_split, valid_split)
    best_rolling_val_acc, j = 0, 0
    hist = {'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'test_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'test_acc': [],
            'bound_value': [],
            'n_sigma': [],
            'n_Z': []}
    if len(weightsbiases) != 1:
        if weightsbiases[0] == 'deeprm2024':
            wandb.login(key='b7d84000aed68a9db76819fa4935778c47f0b374')
        wandb.init(
            name=str(weightsbiases[-1]['start_lr']) + '_' + str(weightsbiases[-1]['optimizer']) + '_' + \
                 str(weightsbiases[-1]['predictor'][-1]) + '_' + str(weightsbiases[-1]['modl_1_dim'][-1]) + '_' + \
                 str(weightsbiases[-1]['k']) + '_' + str(weightsbiases[-1]['seed']),
            project=weightsbiases[1],
            config=weightsbiases[2]
        )
    for i in range(n_epoch):
        meta_pred.train()
        with torch.enable_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                if str(DEVICE) == 'gpu':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                meta_output = meta_pred(inputs)  # We compute the parameters of the predictor.
                output = pred.forward(inputs, meta_output)
                loss = torch.mean(criterion(output, targets)) + penalty_msg(meta_pred.msg,
                                                                            penalty_msg_coef)  # Regularized loss
                loss.backward()
                optimizer.step()
        n_sigma = round(torch.sum((torch.abs(meta_pred.msg) > 0) * 1).item() / batch_size)  # Mean message size
        val_acc, val_loss, _, tr_ac, tr_los = stats(meta_pred, pred, criterion, valid_loader, n_sigma, None, None,
                                                    DEVICE)
        test_acc, test_loss, bound, vd_ac, vd_los = stats(meta_pred, pred, criterion, test_loader, n_sigma, bound_type, None,
                                                      DEVICE)
        train_acc, train_loss, _, te_ac, te_los = stats(meta_pred, pred, criterion, train_loader, n_sigma,None, None, DEVICE) # bound_type, [val_acc, test_acc]
        update_hist(hist, (train_acc, train_loss, val_acc, val_loss, test_acc, test_loss,
                           bound, n_sigma, meta_pred.k, i))  # Tracking results
        rolling_val_acc = torch.mean(torch.tensor(hist['valid_acc'][-min(100,i+1):]))
        if len(weightsbiases) != 1:
            update_wandb(wandb, hist)
        print(f'Epoch {i + 1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f} - Test acc: {test_acc:.2f} - Bound value: {bound:.2f}')
        scheduler.step(rolling_val_acc)
        if i == 1 or rolling_val_acc > best_rolling_val_acc + tol:  # If an improvement has been done in validation...
            j = copy(i)  # ...We keep track of it
            best_rolling_val_acc = copy(rolling_val_acc)
        if (train_acc < 0.525 and i > 50) or i - j > early_stop:  # If no improvement for early_stop epoch, stop training.
            break
        if vis_loss_acc == True and i % 20 == 0:
            if len(weightsbiases) != 1:
                show_loss_acc((tr_los, vd_los, te_los), (tr_ac, vd_ac, te_ac), i, wandb)
            else:
                show_loss_acc((tr_los, vd_los, te_los), (tr_ac, vd_ac, te_ac), i, None)
    if vis > 0 and len(inputs[0, 0, :-1]) == 2:
        if len(weightsbiases) != 1:
            show_decision_boundaries(meta_pred, dataset, test_loader, vis, pred, wandb, DEVICE)
        else:
            show_decision_boundaries(meta_pred, dataset, test_loader, vis, pred, None, DEVICE)

    print()
    if len(weightsbiases) != 1:
        wandb.finish()
    return hist, j


def stats(meta_pred, pred, criterion, data_loader, n_sigma, bound_type, acc, DEVICE):
    """
    Computes the overall accuracy and loss of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (function): A predictor whose parameters are computed by the meta predictor.
        criterion (torch.nn, function): Loss function
        data_loader (DataLoader): A DataLoader to test on.
        n_sigma (int): Size of the message
        bound_type (str): The type of PAC bound to compute (choices: "Alex", "Mathieu").
        DEVICE (str): 'cuda', or 'cpu'; whether to use the gpu
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    meta_pred.eval()
    with torch.no_grad():
        i = 0
        tot_loss = 0
        tot_acc = 0
        for inputs, targets in data_loader:
            n = copy(len(targets) * len(targets[0]))
            i += n
            inputs, targets = inputs.float(), targets.float()
            if str(DEVICE) == 'gpu':
                inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
            meta_output = meta_pred(inputs)
            pred.update_weights(meta_output)
            output = pred.forward(inputs, meta_output, return_sign=True)
            loss = criterion(output[0], targets)
            tot_loss += torch.sum(loss)
            acc = lin_loss(output[1], targets * 2 - 1, reduction=None)
            tot_acc += torch.sum(acc)
            if i == n:
                los = torch.clone(loss)
                ac = torch.clone(acc)
                # out = torch.reshape(output[1], (-1,))
                # tar = torch.reshape(targets, (-1,)) * 2 - 1
            else:
                los = torch.vstack((los, torch.clone(loss)))
                ac = torch.vstack((ac, torch.clone(acc)))
                # out = torch.hstack((out, torch.reshape(output[1], (-1,))))
                # tar = torch.hstack((tar, torch.reshape(targets, (-1,)) * 2 - 1))
        bnd = None
        if bound_type is not None:
            bnd = compute_bound(bound_type=bound_type, n_Z=meta_pred.k, n_sigma=n_sigma,
                                m=i, r=(i - tot_acc).cpu(), c_2=10, d=len(meta_output), acc=acc, delta=0.05)
        return tot_acc / i, tot_loss / i, bnd, ac, los
