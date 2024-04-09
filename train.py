import math
from utils import *
from d2l import torch as d2l
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from copy import copy
from bound import compute_bound
from time import time
import wandb


class SimpleMetaNet(nn.Module):
    def __init__(self, input_dim, dims, output_dim, m, d, tau, batch_size, init, device):
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
        self.ca_dim, self.mha_dim, self.mod_1_dim, self.mod_2_dim = dims  # Dims of all 4 components
        self.ca_dim, self.n_ca = self.ca_dim
        self.k = self.n_ca

        self.msg, self.msg_size, self.msk = None, None, None  # Message and mask (for compression selection)
        self.m, self.d, self.tau, self.batch_size = m, d, tau, batch_size  # Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cas = nn.ModuleList([])
        for i in range(self.n_ca):
            self.cas.append(CA([self.d + 1], self.ca_dim, self.ca_dim, [], self.m, device, init, False, False, 'fspool'))
        self.mha = CA([self.d + 1], self.mha_dim, self.mha_dim, [], self.m, device, init, False, False, 'none')
        self.kme = FSPool([self.d + 1], self.ca_dim, [], self.m, device, init, False, False)
        self.mod_1 = MLP([self.ca_dim[-1]], self.mod_1_dim, [], device, init, False, False)
        self.mod_2 = MLP([self.n_ca ** 2 + self.mod_1_dim[-1]], self.mod_2_dim, [self.output_dim], device, init, False,
                         False)

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input
        return:
            torch.Tensor Output of the network
        """
        # Mask computation #
        if self.n_ca > 0:
            mask = self.cas[0].forward(x.clone())
            for j in range(1, len(self.cas)):
                out = self.cas[j].forward(x.clone())
                mask = torch.hstack((mask, out))

            # Applying the mask to x #
            x_masked = torch.matmul(mask, x.clone())

        # Computing the message #
        x = self.kme.forward(x)
        x = self.mod_1.forward(torch.reshape(x, (len(x), -1)))
        self.msg = x.clone()

        # Computing the compression set description #
        if self.n_ca > 0:
            x_masked = self.mha.forward(x_masked)

            # Concatenating all the information #
            x_masked = torch.reshape(x_masked, (len(x_masked), -1))
            x_red = torch.hstack((x, x_masked))
        else:
            x_red = x

        # Final output computation #
        output = self.mod_2.forward(x_red)
        return output

    def compute_compression_set(self, x):
        # Mask computation #
        if self.n_ca > 0:
            mask = self.cas[0].forward(x.clone())
            for j in range(1, len(self.cas)):
                out = self.cas[j].forward(x.clone())
                mask = torch.hstack((mask, out))
            self.msk = torch.squeeze(torch.topk(mask, 1, dim=(2)).indices)


class CA(nn.Module):
    def __init__(self, input_dim, hidden_dims_mlp, hidden_dims_kme, output_dim, n, device, init, skip, bn, pool):
        super(CA, self).__init__()
        self.k = MLP(input_dim, hidden_dims_mlp, output_dim, device, init, skip, bn)
        if pool == 'kme':
            self.q = KME(input_dim, hidden_dims_kme, output_dim, device, init, skip, bn)
        elif pool == 'fspool':
            self.q = FSPool(input_dim, hidden_dims_kme, output_dim, n, device, init, skip, bn)
        elif pool == 'none':
            self.q = MLP(input_dim, hidden_dims_kme, output_dim, device, init, skip, bn)
        else:
            assert False, 'Wrong pooling choice.'

    def forward(self, x):
        x_1, x_2 = x.clone(), x.clone()
        queries = self.q.forward(x_1)
        keys = self.k.forward(x_2)
        qkt = torch.matmul(queries, torch.transpose(keys, 1, 2))
        dist = torch.softmax(qkt, dim=2)
        return dist


class KME(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, device, init, skip, bn):
        super(KME, self).__init__()
        self.mlp = MLP([input_dim[0] - 1], hidden_dims, output_dim, device, init, skip, bn)

    def forward(self, x):
        batch_size = len(x)
        x_1 = x[:, :, :-1].clone()
        out = self.mlp.forward(x_1)
        x_1 = torch.mean(out * torch.reshape(x[:, :, -1], (batch_size, -1, 1)), dim=1)  # ... A compression is done
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))


class FSPool(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n, device, init, skip, bn):
        super(FSPool, self).__init__()
        self.mlp = MLP([input_dim[0] - 1], hidden_dims, output_dim, device, init, skip, bn)
        self.mat = torch.rand((n, hidden_dims[-1]))

    def forward(self, x):
        batch_size = len(x)
        x_1 = x[:, :, :-1].clone()
        out = self.mlp.forward(x_1)
        out = torch.sort(out, dim=2)[0]
        x_1 = torch.mean(out * self.mat * torch.reshape(x[:, :, -1], (batch_size, -1, 1)), dim=1)  # ... A compression is done
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, device, init, skip, bn):
        super(MLP, self).__init__()
        """
        Creates a ReLU linear layer, given dimensions.
        Args:
            dims (vector of ints): (respectively) size of the input layer, of the hidden layers, and of the output layer
            last (bool): Whether (True) or not (False) to put an activation function at the end of the last layer
        return:
            torch.nn.Module
        """
        self.dims = input_dim + hidden_dims + output_dim
        self.skip = skip
        self.module = torch.nn.ModuleList()
        self.module.to(torch.device(device))
        if self.skip and len(self.dims) > 2:
            self.dims.insert(len(self.dims) - 1, self.dims[0])
        for k in range(len(self.dims) - 1):
            if bn:
                self.module.append(nn.LazyBatchNorm1d())
            self.module.append(nn.Linear(self.dims[k], self.dims[k + 1]))
            if k < len(self.dims) - 2:
                self.module.append(nn.ReLU())

        self.skip_position = len(self.module) - (1 + 1 * bn)
        init_weights(init, self.module)

    def forward(self, x):
        x_1 = x.clone()
        l = 0
        for layer in self.module:
            l += 1
            if self.skip and l == self.skip_position:
                x += x_1
            x = layer(x)
        return x


def init_weights(init, module):
    """
    Instanciate the weights of the linear layers of one or many torch.nn.Module.
    Args:
        init (str): Layer initialization scheme (choices: 'kaiming_unif', '.._norm', 'xavier_unif', '.._norm')
        modules (list of torch.nn.Module): The module(s) of interest
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


def train_valid_loaders(dataset, batch_size, splits, shuffle=True, seed=42):
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
    if len(dataset) > 3:
        num_data = len(dataset)
    else:
        num_data = len(dataset[0])

    indices = np.arange(num_data)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    if len(dataset) > 3:
        split_1 = math.floor(splits[0] * num_data)
        split_2 = math.floor(splits[1] * num_data) + split_1
        train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:split_2], indices[split_2:]
    else:
        split_1 = math.floor(splits[0] / (1 - splits[2]) * num_data)
        train_idx, valid_idx, test_idx = indices[:split_1], indices[split_1:], np.arange(len(dataset[1]))

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def train(meta_pred, pred, data, dataset, splits, train_splits, optimizer, scheduler, tol, early_stop, n_epoch,
          batch_size, criterion, penalty_msg, penalty_msg_coef, vis, vis_loss_acc, bound_type, DEVICE, independent_food,
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
    torch.autograd.set_detect_anomaly(True)
    data_1, data_2 = data
    train_loader_1, valid_loader_1, test_loader_1 = train_valid_loaders(data_1, batch_size, splits)
    train_loader_2, valid_loader_2, test_loader_2 = train_valid_loaders(data_2, batch_size, splits)
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
        wandb.login(key='b7d84000aed68a9db76819fa4935778c47f0b374')
        wandb.init(
            name=str(weightsbiases[-1]['start_lr']) + '_' + str(weightsbiases[-1]['optimizer']) + '_' + \
                 str(weightsbiases[-1]['predictor'][-1]) + '_' + str(weightsbiases[-1]['mod_1_dim'][-1]) + '_' + \
                 str(weightsbiases[-1]['ca_dim'][-1]) + '_' + str(weightsbiases[-1]['seed']),
            project=weightsbiases[1],
            config=weightsbiases[2]
        )
    begin = time()
    for i in range(n_epoch):
        meta_pred.train()
        with torch.enable_grad():
            # k=0
            for inputs_1, targets_1 in train_loader_1:
                #    k+=1
                #    l=0
                #    for inputs_2, targets_2 in train_loader_2:
                #        l+=1
                #        if l != k:
                #            pass
                #        else:
                inputs_1, targets_1 = inputs_1.float(), targets_1.float()
                #    inputs_2, targets_2 = inputs_2.float(), targets_2.float()
                if str(DEVICE) == 'gpu':
                    inputs_1, targets_1, meta_pred = inputs_1.cuda(), targets_1.cuda(), meta_pred.cuda()
                #        inputs_2, targets_2 = inputs_2.cuda(), targets_2.cuda()
                optimizer.zero_grad()
                #    meta_output = meta_pred(inputs_2)  # We compute the parameters of the predictor.
                meta_output = meta_pred(inputs_1)  # We compute the parameters of the predictor.
                pred.update_weights(meta_output)
                output = pred.forward(inputs_1, meta_output)
                loss = torch.mean(torch.mean(criterion(output, targets_1), dim=1) ** 0.5) + penalty_msg(meta_pred.msg,
                                                                              penalty_msg_coef)  # Regularized loss
                loss.backward()
                optimizer.step()
        n_sigma = round(torch.sum((torch.abs(meta_pred.msg) > 0) * 1).item() / batch_size)  # Mean message size
        val_acc, val_loss, _, tr_ac, tr_los = stats(train_splits, meta_pred, pred, criterion, valid_loader_1,
                                                    valid_loader_2, n_sigma, None, None,
                                                    DEVICE, independent_food)
        test_acc, test_loss, bound, vd_ac, vd_los = stats(train_splits, meta_pred, pred, criterion, test_loader_1,
                                                          test_loader_2, n_sigma, bound_type, None,
                                                          DEVICE, independent_food)
        train_acc, train_loss, _, te_ac, te_los = stats(train_splits, meta_pred, pred, criterion, train_loader_1,
                                                        train_loader_2, n_sigma, None, None,
                                                        DEVICE, independent_food)  # bound_type, [val_acc, test_acc]
        update_hist(hist, (train_acc, train_loss, val_acc, val_loss, test_acc, test_loss,
                           bound, n_sigma, meta_pred.k, i))  # Tracking results
        rolling_val_acc = torch.mean(torch.tensor(hist['valid_acc'][-min(100, i + 1):]))
        if len(weightsbiases) != 1:
            update_wandb(wandb, hist)
        print(
            f'Epoch {i + 1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f} - Test acc: {test_acc:.2f} - Bound value: {bound:.2f} - Time (s): {round(time() - begin)}')
        scheduler.step(rolling_val_acc)
        if i == 1 or rolling_val_acc > best_rolling_val_acc + tol:  # If an improvement has been done in validation...
            j = copy(i)  # ...We keep track of it
            best_rolling_val_acc = copy(rolling_val_acc)
        if (
                train_acc < 0.525 and i > 50) or i - j > early_stop:  # If no improvement for early_stop epoch, stop training.
            break
        if vis_loss_acc == True and i % 20 == 0:
            if len(weightsbiases) != 1:
                show_loss_acc((tr_los, vd_los, te_los), (tr_ac, vd_ac, te_ac), i, wandb)
            else:
                show_loss_acc((tr_los, vd_los, te_los), (tr_ac, vd_ac, te_ac), i, None)
    if vis > 0 and len(inputs_1[0, 0, :-1]) == 2:
        if len(weightsbiases) != 1:
            show_decision_boundaries(meta_pred, dataset, test_loader_1, test_loader_2, vis, pred, wandb, DEVICE)

    print()
    if len(weightsbiases) != 1:
        wandb.finish()
    return hist, j


def stats(splits, meta_pred, pred, criterion, data_loader_1, data_loader_2, n_sigma, bound_type, acc, DEVICE,
          independent_food):
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
        k = 0
        for inputs_1, targets_1 in data_loader_1:
            # k += 1
            # l = 0
            # for inputs_2, targets_2 in data_loader_2:
            #    l += 1
            #    if l != k:
            #        pass
            #    else:
            n = copy(len(targets_1) * len(targets_1[0]))
            i += n
            inputs_1, targets_1 = inputs_1.float(), targets_1.float()
            # inputs_2, targets_2 = inputs_2.float(), targets_2.float()
            if str(DEVICE) == 'gpu':
                inputs_1, targets_1, meta_pred = inputs_1.cuda(), targets_1.cuda(), meta_pred.cuda()
            #    inputs_2, targets_2 = inputs_2.cuda(), targets_2.cuda()
            meta_output = meta_pred(inputs_1)  # We compute the parameters of the predictor.
            pred.update_weights(meta_output)
            output = pred.forward(inputs_1, meta_output, return_sign=True)
            loss = criterion(output[0], targets_1)
            tot_loss += torch.sum(loss)
            acc = lin_loss(output[1], targets_1 * 2 - 1, reduction=None)
            tot_acc += torch.sum(acc)
            if i == n:
                los = torch.clone(loss)
                ac = torch.clone(acc)
            else:
                los = torch.vstack((los, torch.clone(loss)))
                ac = torch.vstack((ac, torch.clone(acc)))
        bnd = None
        if bound_type is not None:
            bnd = compute_bound(bound_type=bound_type, n_Z=meta_pred.k, n_sigma=n_sigma,
                                m=i, r=(i - tot_acc).cpu(), c_2=10, d=len(meta_output), acc=acc, delta=0.05)
        return tot_acc / i, tot_loss / i, bnd, ac, los
