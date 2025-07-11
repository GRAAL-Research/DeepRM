import math
import numpy as np
import torch

from src.bound.utils import kl_inv, log_binomial_coefficient, sup_bin, kl_upper_bound
from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.model.utils.loss import linear_loss, linear_loss_multi


def compute_bounds(bnds_type, meta_pred: SimpleMetaNet, pred: Predictor, m, r, delta, a, b, inputs, targets,
                   msg_size: str, msg_type: str, compression_set_size: int):
    """
    Generates the DeepRM meta-predictor.
        Args:
            bnds_type (list of str): all the bounds to be computed (choices: 'kl', 'linear', 'hyperparam', 'marchand',
                                                                                                     'marchand_approx');
            meta_pred (nn.Module): the meta predictor (for the computation of PAC-Bayes bounds);
            m (int): total number of examples;
            r (int): number of error made one the m examples, excluding those made on the compression set;
            delta (float): confidence rate;
            a (float): minimum value for the loss function;
            b (float): maximum value for the loss function;
            inputs (numpy array): examples (for the computation of PAC-Bayes bounds);
            targets (numpy array): labels (for the computation of PAC-Bayes bounds).
        Return:
            list of floats, the bound values.
    """
    n_z = compression_set_size
    n_sigma = msg_size
    msg_std = meta_pred.msg_std
    n_sample, best_bnd, grid_start, grid_stop, n_grid, best_bnds = 100, 0, -5, 5, int(3 * m ** 0.5 / 2), []
    if msg_type == "cnt":  # The bounds are computed differently, depending on the message type
        tot_acc, k = 0, 0  # A Monte-Carlo sampling of messages must be done
        meta_output = meta_pred.forward(inputs, n_noisy_messages=n_sample, is_in_test_mode=True)
        for sample in range(n_sample):
            outp = meta_output[[sample]]
            pred.set_params(outp)
            output = pred.forward(inputs)
            if targets.shape[-1] == 1:
                tot_acc += m * torch.mean(linear_loss(output[1], targets * 2 - 1))
            else:
                tot_acc += m * torch.mean(linear_loss_multi(output[1], targets))
        tot_acc /= n_sample  # An average accuracy is computed...
        r = m - tot_acc
        if meta_pred.get_message().ndim == 0:
            kl = 0
        else:
            kl = torch.mean(torch.sum(meta_pred.get_message() ** 2, dim=1)) / (
                        2 * msg_std)  # ... as well as avg KL val. (shortcut)
        for bnd_type in bnds_type:
            if bnd_type == 'kl': # Corollary D.2 in our article
                epsilon = (kl + log_binomial_coefficient(m, n_z) + np.log(
                    kl_upper_bound(m - n_z) / delta)) / (m - n_z)
                best_bnd = 1 - kl_inv(min((r / (m - n_z)).item(), 1), epsilon.item(), 'MAX')
            elif bnd_type == 'linear': # Corollary D.3 in our article
                for beta in np.logspace(grid_start, grid_stop, n_grid):  # Grid search for the optimal parameter
                    lambd = beta / m ** 0.5
                    bound = 1 - ((r / (m - n_z)) + lambd * (b - a) ** 2 / (8 * (m - n_z)) +
                                 (kl + log_binomial_coefficient(m, n_z) -
                                  np.log(delta / n_grid)) / lambd
                                 ).item()
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'hyperparam': # Corollary D.1 in our article
                for beta in np.logspace(grid_start, grid_stop, n_grid):  # Grid search for the optimal parameter
                    c = beta / m ** 0.5
                    bound = 1 - ((1 - math.exp(-c * (r / (m - n_z)) -
                                               (kl + log_binomial_coefficient(m, n_z) -
                                                np.log(delta / n_grid)) / (m - n_z))) / (1 - math.e ** (-c)))
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'marchand_approx':  # The Marchand-Shaw-Taylor approximation
                best_bnd = 0
            elif bnd_type == 'marchand':
                best_bnd = 0
            elif bnd_type == 'kl_dis': # Corollary F.1 in our article
                epsilon = (2 * kl + log_binomial_coefficient(m, n_z) + np.log(
                    kl_upper_bound(m - n_z)) + 3 * np.log(2 / delta)) / (m - n_z)
                best_bnd = 1 - kl_inv(min((r / (m - n_z)).item(), 1), epsilon.item(), 'MAX')
            best_bnds.append(best_bnd)
    elif msg_type == "dsc":
        for bnd_type in bnds_type:
            p_sigma = 2 ** (-n_sigma)  # Since the message is a binary vector, we consider a uniform distribution
            #   on its various possibilities (prob = 2 ** -number of possibilities)
            if bnd_type == 'kl': # Corollary A.2 in our article
                epsilon = (log_binomial_coefficient(m, n_z) +
                           np.log(kl_upper_bound(m - n_z) / p_sigma / delta)) / (m - n_z)
                best_bnd = 1 - kl_inv(min(1, r / (m - n_z)), epsilon, 'MAX')
            elif bnd_type == 'linear': # Corollary A.3 in our article
                for beta in np.logspace(grid_start, grid_stop, n_grid):  # Grid search for the optimal parameter
                    lambd = beta / m ** 0.5
                    bound = 1 - ((r / (m - n_z)) + lambd * (b - a) ** 2 / 8 +
                                 (log_binomial_coefficient(m, n_z) -
                                  np.log(p_sigma) -
                                  np.log(delta / n_grid)) / (lambd * (m - n_z))
                                 )
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'hyperparam': # Corollary A.1 in our article
                for beta in np.logspace(grid_start, grid_stop, n_grid):  # Grid search for the optimal parameter
                    c = beta / m ** 0.5
                    bound = 1 - ((1 - math.exp(-c * (r / (m - n_z)) -
                                               (log_binomial_coefficient(m, n_z) -
                                                np.log(p_sigma) -
                                                np.log(delta / n_grid)) / (m - n_z)))
                                 / (1 - math.exp(-c))
                                 )
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'marchand_approx':  # The Marchand-Shaw-Taylor approximation
                best_bnd = math.exp((-1 / (m - r - n_z)) * (
                        log_binomial_coefficient(m - n_z, r) -
                        np.log(p_sigma) -
                        np.log(delta)))
            elif bnd_type == 'marchand':
                best_bnd = 1 - sup_bin(int(min(r, m - n_z)), int(m - n_z),  # The test-set bound for sample compression
                                       delta * p_sigma / math.exp(log_binomial_coefficient(m, n_z)))
            elif bnd_type == 'kl_dis':
                best_bnd = 0
            best_bnds.append(best_bnd)
    return best_bnds
