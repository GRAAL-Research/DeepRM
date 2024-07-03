import math

import numpy as np
import torch

from src.bound.utils import zeta, kl_inv, log_binomial_coefficient, sup_bin
from src.model.utils.loss import lin_loss


def compute_bound(bnds_type, meta_pred, pred, m, r, delta, a, b, inputs, targets):
    """
    Generates the DeepRM meta-predictor.
        Args:
            bnds_type (list of str): all the bounds to be computed (choices: 'kl', 'linear', 'hyperparam', 'marchand',
                                                                                                     'marchand_approx');
            meta_pred (nn.Module): the meta predictor (for the computation of PAC-Bayes bounds);
            pred (nn.Module): the predictor (for the computation of PAC-Bayes bounds);
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
    n_z = meta_pred.comp_set_size
    n_sigma = meta_pred.msg_size
    msg_type = meta_pred.msg_type
    n_sample, best_bnd, n_grid, best_bnds = 2, 0, 11, []
    for bnd_type in bnds_type:
        if msg_type == 'cnt':  # The bounds are calculated differently, depending on the message type
            tot_acc, k = 0, 0  # A Monte-Carlo sampling of messages must be done
            meta_output = meta_pred(inputs, n_samples=n_sample)
            for sample in range(n_sample):
                outp = meta_output[[sample]]
                pred.update_weights(outp, 1)
                output = pred.forward(inputs, return_sign=True)
                tot_acc += m * lin_loss(output[1], targets * 2 - 1)
            tot_acc /= n_sample  # An average accuracy is computed...
            r = m - tot_acc
            kl = torch.mean(torch.sum(meta_pred.msg ** 2, dim=1))  # ... as well as an avg KL value (shortcut, lighter)
            if bnd_type == 'kl':
                epsilon = (kl + np.log(2 * np.sqrt(m - n_z) / zeta(n_z) / delta)) / (m - n_z)
                best_bnd = 1 - kl_inv(min((r / (m - n_z)).item(), 1), epsilon.item(), 'MAX')
            elif bnd_type == 'linear':
                grid_start = -5
                for beta in np.logspace(grid_start, grid_start + n_grid):  # Grid search for the optimal parameter
                    lambd = beta / m ** 0.5
                    bound = 1 - ((r / (m - n_z)) + lambd * (b - a) ** 2 / (8 * (m - n_z)) +
                                 (kl -
                                  np.log(zeta(n_z)) -
                                  np.log(delta / n_grid)) / lambd
                                 ).item()
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'hyperparam':
                grid_start = -5
                for beta in np.logspace(grid_start, grid_start + n_grid):  # Grid search for the optimal parameter
                    c = beta / m ** 0.5
                    bound = 1 - ((1 - math.exp(-c * (r / (m - n_z)) -
                                               (kl -
                                                np.log(zeta(n_z)) -
                                                np.log(delta / n_grid)) / (m - n_z))) / (1 - math.e ** (-c)))
                    if bound > best_bnd:
                        best_bnd = bound
            elif 'marchand' in bnd_type:  # The marchand bound cannot be computed with continuous messages
                best_bnd = 0
        elif msg_type == 'dsc':
            p_sigma = 2 ** (-n_sigma)  # Since the message is a binary vector, we consider a uniform distribution
            #   on its various possibilities (prob = 2 ** -number of possibilities)
            if bnd_type == 'kl':
                epsilon = (log_binomial_coefficient(m, n_z) +
                           np.log(2 * np.sqrt(m - n_z) / zeta(n_z) / p_sigma / delta)) / (m - n_z)
                best_bnd = 1 - kl_inv(min(1, r / (m - n_z)), epsilon, 'MAX')
            elif bnd_type == 'linear':
                grid_start = -5
                for beta in np.logspace(grid_start, grid_start + n_grid):  # Grid search for the optimal parameter
                    lambd = beta / m ** 0.5
                    bound = 1 - ((r / (m - n_z)) + lambd * (b - a) ** 2 / 8 +
                                 (log_binomial_coefficient(m, n_z) -
                                  np.log(p_sigma) -
                                  np.log(zeta(n_z)) -
                                  np.log(delta / n_grid)) / (lambd * (m - n_z))
                                 )
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'hyperparam':
                grid_start = -5
                for beta in np.logspace(grid_start, grid_start + n_grid):  # Grid search for the optimal parameter
                    c = beta / m ** 0.5
                    bound = 1 - ((1 - math.exp(-c * (r / (m - n_z)) -
                                               (log_binomial_coefficient(m, n_z) -
                                                np.log(p_sigma) -
                                                np.log(zeta(n_z)) -
                                                np.log(delta / n_grid)) / (m - n_z)))
                                 / (1 - math.exp(-c))
                                 )
                    if bound > best_bnd:
                        best_bnd = bound
            elif bnd_type == 'marchand':
                best_bnd = 1 - sup_bin(int(min(r, m - n_z)), int(m - n_z),  # The test-set bound for sample compression
                                       delta * p_sigma * zeta(n_z) / math.exp(log_binomial_coefficient(m, n_z)))
            elif bnd_type == 'marchand_approx':  # The Marchand-Shaw-Taylor approximation
                best_bnd = math.exp((-1 / (m - r - n_z)) * (
                        log_binomial_coefficient(m - n_z, r) -
                        np.log(p_sigma) -
                        np.log(zeta(n_z)) -
                        np.log(delta)
                )
                                    )
        best_bnds.append(best_bnd)
    return best_bnds
