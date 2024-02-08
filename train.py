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
        self.kern_1_dim, self.kern_2_dim, self.modl_1_dim, self.modl_2_dim, self.modl_3_dim, self.modl_4_dim = dims  # Dims of all 5 components
        self.msg, self.msg_size, self.msk = None, self.modl_3_dim[-1], None  # Message and mask (for compression selection)
        self.m, self.d, self.k, self.tau, self.batch_size = m, d, k, tau, batch_size  # Parameters
        self.inds = torch.zeros((self.batch_size, self.k), dtype=int)
        self.output_dim = output_dim
        assert 0 <= k <= m # Sample compression size is bounded
        self.output_dim_3 = self.modl_3_dim[-1] # So that input size is (message size + mask size)
        self.output_dim_4 = output_dim
        self.modl_3_dim[-1] = self.kern_1_dim[-1]
        self.modl_4_dim[-1] = self.output_dim_3 + self.kern_2_dim[-1]

        self.modl_1 = relu_fc_layer([self.d + 1] + self.modl_1_dim, self.device)
        self.modl_2 = relu_fc_layer([self.d + 1] + self.modl_2_dim, self.device)
        self.modl_3 = relu_fc_layer([self.kern_1_dim[-1]] + self.modl_3_dim, self.device, last = True)
        self.modl_4 = relu_fc_layer([self.output_dim_3 + self.kern_2_dim[-1]] + self.modl_4_dim, self.device, last = True)
        self.last_l_3 = relu_fc_layer([self.modl_3_dim[-1]] + [self.output_dim_3], self.device)
        self.last_l_4 = relu_fc_layer([self.modl_4_dim[-1]] + [self.output_dim_4], self.device)
        self.kern_1 = relu_fc_layer([input_dim] + self.kern_1_dim, self.device)  # The different layers are instanciated
        self.kern_2 = relu_fc_layer([self.d] + self.kern_2_dim, self.device)
        init_weights_ff(init, [self.modl_1, self.modl_2, self.modl_3, self.last_l_3, self.modl_4, self.last_l_4, self.kern_1, self.kern_2])

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input
        return:
            torch.tensor Output of the network
        """
        x_1, x_2 = x, x

        ## First module ##
        for layer in self.modl_1:
            x_1 = layer(x_1)

        ## Second module ##
        for layer in self.modl_2:
            x_2 = layer(x_2)

        ## Mask computation ##
        x_t = torch.sum(torch.mul(x_1, x_2), dim=-1) * self.tau
        x_t = torch.softmax(x_t, dim=1).clone()

        self.probs = torch.clone(x_t)
        if int(self.k) == self.k:
            odds = F.gumbel_softmax(torch.log(x_t+1e-40), hard=False)
            self.inds = torch.topk(odds, self.k).indices  # We retain the top-k examples having the highest odds
            tens = torch.zeros(odds.shape).to(torch.device(self.device))
            tens[torch.arange(tens.size(0)).unsqueeze(1), self.inds] = 1
            tens += odds - odds.detach()
        else:
            odds = F.gumbel_softmax(torch.log(x_t), hard=False)
            self.inds = (odds > self.k)  # We retain the examples having higher odds than k
            tens = torch.zeros(odds.shape).to(torch.device(self.device))
            tens[torch.arange(tens.size(0)).unsqueeze(1), self.inds] = 1
            tens += odds - odds.detach()
        self.msk = torch.clone(tens)

        ## First KME, all data ##
        x_2 = x[:, :, :-1]
        for layer in self.kern_1:
            x_2 = layer(x_2)
        x_1 = torch.mean(x_2 * torch.reshape(x[:, :, -1], (self.batch_size, -1, 1)), dim=1)  # ... A compression is done

        ## First KME, masked data ##
        x_2 = x[:, :, :-1] * torch.reshape(tens, (self.batch_size, -1, 1))
        if self.k > 0:
            for layer in self.kern_1:
                x_2 = layer(x_2)
            x_2 = torch.mean(x_2 * torch.reshape(x[:, :, -1], (self.batch_size, -1, 1)), dim=1)  # ... A compression is done
        else:
            x_2 = torch.zeros((self.batch_size, self.kern_1_dim[-1]))

        ## Third module ##
        x_t = x_1 - x_2
        skip = x_t  # A skip connection is made
        for layer in self.modl_3:
            x_t = layer(x_t)
        for layer in self.last_l_3:
            x_t = layer(x_t + skip)  # The skip connection leads to just before applying the last linear layer
        self.msg = x_t

        ## Second KME ##
        x_1 = x[:, :, :-1] * torch.reshape(tens, (self.batch_size, -1, 1))
        if self.k > 0:
            for layer in self.kern_2:
                x_1 = layer(x_1)
            x_1 = torch.mean(x_1 * torch.reshape(x[:, :, -1], (self.batch_size, -1, 1)), dim=1)  # ... A compression is done
        else:
            x_1 = torch.zeros((self.batch_size, self.kern_2_dim[-1]))

        ## Fourth module ##
        x_2 = torch.hstack((x_1, x_t))
        skip = x_2  # A skip connection is made
        for layer in self.modl_4:
            x_2 = layer(x_2)
        for layer in self.last_l_4:
            x_2 = layer(x_2 + skip)  # The skip connection leads to just before applying the last linear layer
        return x_2

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

def attention_layer(embed_dim, num_heads, device):
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
    lay.append({'attention':nn.MultiheadAttention(embed_dim, num_heads),
                'query':torch.tensor(embed_dim),
                'key':torch.tensor(embed_dim),
                'values':torch.tensor(embed_dim)})
    return lay

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = torch.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

def init_weights_ff(init, modules):
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

def init_weights_attention(init, modules):
    """
        Instanciate the weights of the linear layers of one or many torch.nn.Module.
        Args:
            init (str): Layer initialization scheme (choices: 'kaiming_unif', '.._norm', 'xavier_unif', '.._norm')
            modules (list of torch.nn.Module): The module(s) of interest
        """

    for module in modules:
        for layer in module:
            pass


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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
    train_loader, valid_loader, test_loader = train_valid_loaders(data, batch_size, splits)
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
                 str(weightsbiases[-1]['predictor'][-1]) + '_' + str(weightsbiases[-1]['modl_1_dim'][-1]) + '_' + \
                 str(weightsbiases[-1]['k']) + '_' + str(weightsbiases[-1]['seed']),
            project=weightsbiases[1],
            config=weightsbiases[2]
        )
    begin = time()
    for i in range(n_epoch):
        meta_pred.train()
        with torch.enable_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                if str(DEVICE) == 'gpu':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                if independent_food:
                    split = math.floor(train_splits[0] * len(inputs[0]))
                    inputs_1, inputs_2 = inputs[:,:split], inputs[:,split:]
                    indices = np.arange(split)
                    np.random.shuffle(indices)
                    indices = indices[:train_splits[1]]
                    inputs_1 = inputs_1[:,indices]
                    targets_2 = targets[:,split:]
                else:
                    inputs_1, inputs_2, targets_2 = inputs, inputs, targets
                optimizer.zero_grad()
                meta_output = meta_pred(inputs_1)  # We compute the parameters of the predictor.
                pred.update_weights(meta_output)
                output = pred.forward(inputs_2, meta_output)
                loss = torch.mean(criterion(output, targets_2)) + penalty_msg(meta_pred.msg,
                                                                            penalty_msg_coef)  # Regularized loss
                loss.backward()
                optimizer.step()
        n_sigma = round(torch.sum((torch.abs(meta_pred.msg) > 0) * 1).item() / batch_size)  # Mean message size
        val_acc, val_loss, _, tr_ac, tr_los = stats(train_splits, meta_pred, pred, criterion, valid_loader, n_sigma, None, None,
                                                    DEVICE, independent_food)
        test_acc, test_loss, bound, vd_ac, vd_los = stats(train_splits, meta_pred, pred, criterion, test_loader, n_sigma, bound_type, None,
                                                      DEVICE, independent_food)
        train_acc, train_loss, _, te_ac, te_los = stats(train_splits, meta_pred, pred, criterion, train_loader, n_sigma,None, None,
                                                        DEVICE, independent_food) # bound_type, [val_acc, test_acc]
        update_hist(hist, (train_acc, train_loss, val_acc, val_loss, test_acc, test_loss,
                           bound, n_sigma, meta_pred.k, i))  # Tracking results
        rolling_val_acc = torch.mean(torch.tensor(hist['valid_acc'][-min(100,i+1):]))
        if len(weightsbiases) != 1:
            update_wandb(wandb, hist)
        print(f'Epoch {i + 1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f} - Test acc: {test_acc:.2f} - Bound value: {bound:.2f} - Time (s): {round(time()-begin)}')
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


def stats(splits, meta_pred, pred, criterion, data_loader, n_sigma, bound_type, acc, DEVICE, independent_food):
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
            if independent_food:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                indices = indices[:splits[1]]
                inputs_1 = inputs[:, indices]
            else:
                inputs_1 = inputs
            meta_output = meta_pred(inputs_1)  # We compute the parameters of the predictor.
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
