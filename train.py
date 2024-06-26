import math
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from copy import copy
from bound import compute_bound
from time import time
import wandb


class SimpleMetaNet(nn.Module):
    def __init__(self, pred_input_dim, task_dict):
        """
        Generates the DeepRM meta-predictor.
        Args:
            pred_input_dim (int): Input dimension of the predictor;
            task_dict (dictionary) containing the following:
                m (int): Number of examples per dataset;
                d (int): Input dimension of each dataset;
                comp_set_size (int): compression set size;
                msg_size (int): message size;
                ca_dim (list of int): custom attention's MLP architecture;
                kme_dim (list of int): KME's MLP architecture;
                mod_1_dim (list of int): MLP #1 architecture;
                mod_2_dim (list of int): MLP #2 architecture;
                tau (int): temperature parameter (softmax in custom attention);
                msg_type (str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
                batch_size (int): Batch size.
                init (str): rand. init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
                device (str): device on which to compute (choices: 'cpu', 'gpu');
        """
        super(SimpleMetaNet, self).__init__()
        # Saving the parameters of the meta-learner
        self.device = task_dict['device']
        self.comp_set_size, self.msg_size = task_dict['comp_set_size'], task_dict['msg_size']
        self.ca_dim, self.kme_dim = task_dict['ca_dim'], task_dict['kme_dim']
        self.mod_1_dim, self.mod_2_dim = task_dict['mod_1_dim'], task_dict['mod_2_dim']
        self.m, self.msg_type, self.init = int(task_dict['m'] / 2), task_dict['msg_type'], task_dict['init']
        self.msg, self.msk = torch.tensor(0.0), None  # Message and mask (compression selection)
        self.msg_size = task_dict['msg_size']
        self.d, self.tau, self.batch_size = task_dict['d'], task_dict['tau'], task_dict['batch_size']  # Parameters
        self.input_dim = self.d
        self.output_dim = pred_input_dim
        self.mod_2_input = self.kme_dim[-1] * (self.comp_set_size > 0) + self.mod_1_dim[-1] * (self.msg_size > 0)

        # Generating the many components (custom attention (CA) multi-heads, KME #1-2, MLP #1-2) of the meta-learner
        self.cas = nn.ModuleList([])
        for i in range(self.comp_set_size):
            self.cas.append(CA(self.d + 1, self.ca_dim, self.ca_dim, self.m, self.device,
                               self.init, False, False, 'fspool', self.tau))
        self.kme_1 = KME(self.d + 1, self.ca_dim, self.device, self.init, False, False)
        self.kme_2 = KME(self.d + 1, self.kme_dim, self.device, self.init, False, False)
        self.mod_1 = MLP(self.ca_dim[-1], self.mod_1_dim, self.device, self.init,
                         False, False, self.msg_type)
        self.mod_2 = MLP(self.mod_2_input, self.mod_2_dim + [self.output_dim], self.device,
                         self.init, False, False, 'cnt')

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

            if self.msg_type == 'cnt':
                x = x * 3   # See bound computation
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
            assert False, 'Cannot compute the compression set when it is of size 0.'


class CA(nn.Module):
    def __init__(self, input_dim, hidden_dims_mlp, hidden_dims_kme, m, device, init, skip, bn, pool, temp):
        """
        Initialize a custom attention head.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims_mlp (list of int): architecture of the MLP;
            hidden_dims_kme (list of int): architecture of the embedding (MLP) in the KME;
            m (int): number of examples per dataset;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not;
            pool (str): type of pooling to apply for the query computation (choices: 'kme', 'fspool', 'none');
            temp (float): temperature parameter for the softmax computation.
        """
        super(CA, self).__init__()
        self.temp = temp
        #   The Keys are always computed by an MLP...
        self.k = MLP(input_dim, hidden_dims_mlp, device, init, skip, bn, 'cnt')
        #   While the Queries might be the result of a pooling component.
        if pool == 'kme':
            self.q = KME(input_dim, hidden_dims_kme, device, init, skip, bn)
        elif pool == 'fspool':
            self.q = FSPool(input_dim, hidden_dims_kme, m, device, init, skip, bn)
        elif pool == 'none':
            self.q = MLP(input_dim, hidden_dims_kme, device, init, skip, bn, 'cnt')
        else:
            assert False, 'Wrong pooling choice.'

    def forward(self, x):
        """
        Computes a forward pass, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1, x_2 = x.clone(), x.clone()
        queries = self.q.forward(x_1)
        keys = self.k.forward(x_2)
        qkt = torch.matmul(queries, torch.transpose(keys, 1, 2))
        dist = torch.softmax(self.temp * qkt / torch.max(qkt, dim=-1).values, dim=2)
        return dist


class KME(nn.Module):
    def __init__(self, input_dim, hidden_dims, device, init, skip, bn):
        """
        Initialize a custom attention head.
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the embedding;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not.
        """
        super(KME, self).__init__()
        self.embedding = MLP(input_dim - 1, hidden_dims, device, init, skip, bn, 'cnt')

    def forward(self, x):
        """
        Computes the KME output, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1 = x[:, :, :-1].clone()
        out = self.embedding.forward(x_1)
        x_1 = torch.mean(out * torch.reshape(x[:, :, -1], (len(x), -1, 1)), dim=1)  # ... A compression is done
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))


class FSPool(nn.Module):
    def __init__(self, input_dim, hidden_dims, m, device, init, skip, bn):
        """
        Initialize .
        Args:
            input_dim (int): input dimension of the custom attention head;
            hidden_dims (list of int): architecture of the embedding;
            m (int): number of examples per dataset;
            device (str): device on which to compute (choices: 'cpu', 'gpu');
            init (str): random init. (choices: 'kaiming_unif', 'kaiming_norm', 'xavier_unif', 'xavier_norm');
            skip (bool): whether to include a skip connection or not;
            bn (bool): whether to include batch normalization or not.
        """
        super(FSPool, self).__init__()
        self.mlp = MLP(input_dim - 1, hidden_dims, device, init, skip, bn, 'cnt')
        self.mat = torch.rand((m, hidden_dims[-1]))
        if device == 'gpu':
            self.mat = self.mat.to('cuda:0')

    def forward(self, x):
        """
        Computes FSPool output, given an input.
        Args:
            x (torch.tensor of floats): input;
        return:
            torch.Tensor: output of the custom attention heads layer.
        """
        x_1 = x[:, :, :-1].clone()
        out = self.mlp.forward(x_1)
        out = torch.sort(out, dim=2)[0]

        x_1 = torch.mean(out * self.mat * torch.reshape(x[:, :, -1], (len(x), -1, 1)), dim=1)
        return torch.reshape(x_1, (x_1.shape[0], 1, -1))


def train_valid_loaders(dataset, batch_size, splits, shuffle=True, seed=42):
    """
    Divides a dataset into a training set and a validation set, both in a Pytorch DataLoader form.
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Desired batch-size for the DataLoader
        splits (list of float): Desired proportion of training, validation and test example must sum to 1).
        shuffle (bool): Whether the examples are shuffled before train/validation split.
        seed (int): A random seed.
    Returns:
        Tuple (training DataLoader, validation DataLoader).
    """
    assert sum(splits) == 1, 'The sum of splits must be 1.'
    num_data = len(dataset)
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


def train(meta_pred, pred, data, optimizer, scheduler, criterion, pen_msg, task_dict):
    """
    Trains a meta predictor using PyTorch.

    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (Predictor): A predictor whose parameters are computed by the meta predictor.
        data (Dataset): A dataset.
        optimizer (torch.optim): meta-neural network optimizer;
        scheduler (torch.optim.lr_scheduler): learning rate decay scheduler;
        criterion (function): loss function (choices: 'bce_loss');
        pen_msg (function): message penalty function;
        task_dict (dictionary) containing the following:
            splits ([float, float, float]): train, valid and test proportion of the data;
            tol (float): Quantity by which the loss must diminish in order for this increment not to be marginal
            early_stop (int): Number of epochs by which, if the loss hasn't diminished by « tol », the train stops
            n_epoch (int): The maximum number of epochs.
            batch_size (int): Batch size.
            pen_msg_coef (float): Message regularization factor.
            device (str): whether to use the gpu (choices: 'gpu', 'cpu');
            weightsbiases (list of [str, str]): list with WandB team and project if data is to be stocked on WandB;
                          (empty list): if data is not to be stocked in WandB.
    Returns:
        tuple of: information about the model at the best training epoch (dictionary), best training epoch (int).
    """
    # Retrieving information
    splits = task_dict['splits']
    tol = task_dict['tol']
    early_stop = task_dict['early_stop']
    n_epoch = task_dict['n_epoch']
    batch_size = task_dict['batch_size']
    msg_size = task_dict['msg_size']
    msg_type = task_dict['msg_type']
    pen_msg_coef = task_dict['pen_msg_coef']
    device = task_dict['device']
    weightsbiases = task_dict['weightsbiases']
    m = meta_pred.m

    torch.autograd.set_detect_anomaly(True)
    train_loader, valid_loader, test_loader = train_valid_loaders(data, batch_size, splits)
    best_rolling_val_acc, best_epoch = 0, 0
    # The following information will be recorded at each epoch
    hist = {'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'test_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'test_acc': [],
            'bound_lin': [],
            'bound_hyp': [],
            'bound_kl': [],
            'bound_mrch': []}
    if len(weightsbiases) > 1:  # If condition is true, then metrics about the experiment are recorded in WandB
        wandb.login(key='b7d84000aed68a9db76819fa4935778c47f0b374')
        wandb.init(
            name=str(task_dict['start_lr']) + '_' + str(task_dict['optimizer']) + '_' +
            str(task_dict['msg_type']) + '_' + str(task_dict['pred_arch']) + '_' +
            str(task_dict['msg_size']) + '_' + str(task_dict['comp_set_size']) + '_' +
            str(task_dict['pen_msg_coef']) + '_' + str(task_dict['seed']),
            project=weightsbiases[1],
            config=task_dict
        )
    begin = time()
    for i in range(n_epoch):
        meta_pred.train()  # We put the meta predictor in training mode
        with (torch.enable_grad()):
            for inputs in train_loader:  # Iterating over the various batches
                targets = (inputs.clone()[:, :, -1] + 1) / 2
                inputs, targets = inputs.float(), targets.float()
                if str(device) == 'gpu':
                    inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
                optimizer.zero_grad()  # Zeroing the gradient everywhere in the meta-learner
                meta_output = meta_pred(inputs[:, m:])  # Computing the parameters of the predictor.
                pred.update_weights(meta_output, len(inputs))  # Updating the weights of the predictor
                output = pred.forward(inputs[:, m:])  # Computing the predictions for the task
                loss = torch.mean(torch.mean(criterion(output, targets[:, m:]), dim=1) ** 0.5)
                loss += pen_msg(meta_pred.msg, pen_msg_coef)  # Regularized loss
                loss.backward()  # Gradient computation
                optimizer.step()  # Backprop step
        # Computation of statistics about the current training epoch
        tr_acc, tr_loss, _ = stats(meta_pred, pred, criterion, train_loader, msg_type, device)
        vd_acc, vd_loss, _ = stats(meta_pred, pred, criterion, valid_loader, msg_type, device)
        te_acc, te_loss, bound = stats(meta_pred, pred, criterion, test_loader, msg_type, device)
        update_hist(hist, (tr_acc, tr_loss, vd_acc, vd_loss, te_acc, te_loss, bound, msg_size,
                           meta_pred.comp_set_size, i))  # Tracking results
        rolling_val_acc = torch.mean(torch.tensor(hist['valid_acc'][-min(100, i + 1):]))
        if len(weightsbiases) > 1:
            update_wandb(wandb, hist)  # Upload information to WandB
        epo = '0' * (i + 1 < 100) + '0' * (i + 1 < 10) + str(i+1)
        print(f'Epoch {epo} - Train acc: {tr_acc:.2f} - Val acc: {vd_acc:.2f} - Test acc: {te_acc:.2f} - '
              f'Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), '
              f'(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - begin)}')  # Print information in console
        scheduler.step(rolling_val_acc)  # Scheduler step
        if i == 1 or rolling_val_acc > best_rolling_val_acc + tol:  # If an improvement has been done in validation...
            best_epoch = copy(i)  # ...We keep track of it
            best_rolling_val_acc = copy(rolling_val_acc)
        if ((tr_acc < 0.525 and i > 50) or  # If no learning has been made...
                i - best_epoch > early_stop):  # ... or no improvements for a while ...
            break  # Early stopping is made
    if task_dict['d'] == 2 and len(weightsbiases) > 1:  # Plotting the decision boundary for a few problems
        show_decision_boundaries(meta_pred, task_dict['dataset'], test_loader, pred, wandb, device)
    if len(weightsbiases) > 1:
        wandb.finish()  # End the run on WandB
    return hist, best_epoch


def stats(meta_pred, pred, criterion, data_loader, msg_type, device):
    """
    Computes the overall accuracy, loss and bounds of a predictor on given task and dataset.
    Args:
        meta_pred (nn.Module): A meta predictor (neural network) to train.
        pred (Predictor): A predictor whose parameters are computed by the meta predictor.
        criterion (torch.nn, function): Loss function
        data_loader (DataLoader): A DataLoader to test on.
        msg_type (str): type of message (choices: 'dsc' (discrete), 'cnt' (continuous));
        device (str): 'cuda', or 'cpu'; whether to use the gpu
    Returns:
        Tuple (float, float): the 0-1 accuracy and loss.
    """
    bnd_lin, bnd_hyp, bnd_kl, bnd_mrch = [], [], [], []  # The various bounds to compute
    meta_pred.eval()  # We put the meta predictor in evaluation mode
    m = meta_pred.m
    with torch.no_grad():
        i, k = 0, 0  # Number of batches / examples we have been through
        tot_loss, tot_acc = [], []
        for inputs in data_loader:
            targets = (inputs.clone()[:, :, -1] + 1) / 2
            n = copy(len(targets) * len(targets[0]))
            i += n
            k += 1
            inputs, targets = inputs.float(), targets.float()
            if str(device) == 'gpu':
                inputs, targets, meta_pred = inputs.cuda(), targets.cuda(), meta_pred.cuda()
            meta_output = meta_pred(inputs[:, :m])  # Computing the parameters of the predictor
            pred.update_weights(meta_output, len(inputs))  # Updating the weights of the predictor
            output = pred.forward(inputs[:, m:], True)  # Computing the predictions for the task
            loss = criterion(output[0], targets[:, m:])  # Loss computation
            tot_loss.append(torch.sum(loss).cpu())
            acc = m * lin_loss(output[1], targets[:, m:] * 2 - 1)  # Accuracy computation
            tot_acc.append(torch.mean(acc / m).cpu())
            if msg_type is not None:
                for b in range(len(inputs)):  # For all datasets, we compute the bounds
                    bnds = compute_bound(['linear', 'hyperparam', 'kl', 'marchand'], meta_pred, pred, m,
                                         m - acc.item(), 0.05, 0, 1, inputs[[b], m:], targets[[b], m:])
                    bnd_lin.append(bnds[0])
                    bnd_hyp.append(bnds[1])
                    bnd_kl.append(bnds[2])
                    bnd_mrch.append(bnds[3])
        if msg_type is None:
            return np.mean(tot_acc), np.mean(tot_loss), []
        # We only return the mean bound obtained on the various datasets
        return np.mean(tot_acc), np.mean(tot_loss), [np.mean(bnd_lin), np.mean(bnd_hyp), np.mean(bnd_kl),
                                                     np.mean(bnd_mrch)]
