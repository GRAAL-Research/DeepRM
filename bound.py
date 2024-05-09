import math
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch
from utils import *


def log_stirlings_approximation(n):
    """
    Stirling's approximation for the logarithm of the factorial

    """
    if n == 0:
        return 0
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


def log_binomial_coefficient(n, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation
    """
    return (log_stirlings_approximation(n) -
            log_stirlings_approximation(k) -
            log_stirlings_approximation(n - k))


def log_prob_bin(k, n, r):
    return log_binomial_coefficient(n, k) + k * math.log(max(r, 1e-10)) + (n - k) * math.log(max(1 - r, 1e-10))


def bin_cum(k, n, r):
    prob_cum = 0
    for i in range(k + 1):
        prob_cum += math.exp(log_prob_bin(i, n, r))
    return prob_cum


def sup_bin(k, m, delta):
    gamma_sup = 1
    gamma_inf = 0
    for j in range(10):
        gamma = (gamma_sup + gamma_inf) / 2
        pro = bin_cum(k, m, gamma)
        if pro >= delta:
            gamma_inf = gamma
        else:
            gamma_sup = gamma
    return gamma


def inf_bin(k, m, delta):
    gamma_sup = 1
    gamma_inf = 0
    for j in range(10):
        gamma = (gamma_sup + gamma_inf) / 2
        pro = bin_cum(k, m, gamma)
        if 1 - pro >= delta:
            gamma_sup = gamma
        else:
            gamma_inf = gamma
    return gamma


def zeta(x):
    """
    Mario's function...
    """
    return (6 / np.pi ** 2) * (x + 1) ** -2


def eta(x, m):
    """
    Uniform
    """
    return 1 / m


def prior(sigma):
    """
    Prior on the value of messages (this is an arbitrary choice)

    """
    # Length prior: we make an arbitrary choice that decays with message length (favor smaller messages)
    p_len = zeta(len(sigma))

    # Content prior: we use a simple isotropic standard normal prior (favors small weights)
    p_content = np.prod(norm.pdf(sigma))  # Product of likelihood of each dimension

    return p_len * p_content


def a_star(m, n_Z, r, a):
    return (1 -
            np.exp(1 / (m - n_Z - r) * (
                        r * np.log(r / (m - n_Z)) + (m - n_Z - r) * np.log(1 - r / (m - n_Z)) - r * np.log(
                    r / (m - n_Z) + a))) -
            r / (m - n_Z))


def bnd(m, n_Z, r, p_sigma, delta, a):
    return 1 - np.exp(
        -1 / (m - n_Z - r) *
        (r * np.log(r / (m - n_Z) + a) +
         log_binomial_coefficient(m, n_Z) +
         log_binomial_coefficient(m - n_Z, r) +
         np.log(1 / p_sigma) +
         np.log(1 / (zeta(n_Z) * zeta(r) * delta))
         ))


def kl_inv(q, epsilon, mode, tol=10 ** -9, nb_iter_max=1000):
    """
    Solve the optimization problem min{ p in [0, 1] | kl(q||p) <= epsilon }
    or max{ p in [0,1] | kl(q||p) <= epsilon } for q and epsilon fixed

    Parameters
    ----------
    q: float
        The parameter q of the kl divergence
    epsilon: float
        The upper bound on the kl divergence
    tol: float, optional
        The precision tolerance of the solution
    nb_iter_max: int, optinal
        The maximum number of iterations
    """
    assert mode == "MIN" or mode == "MAX"
    assert q >= 0 and q <= 1, f"q is out of bounds: must be within [0,1], has value {q}."
    assert isinstance(epsilon, float) and epsilon > 0.0

    def kl(q, p):
        """
        Compute the KL divergence between two Bernoulli distributions
        (denoted kl divergence)

        Parameters
        ----------
        q: float
            The parameter of the posterior Bernoulli distribution
        p: float
            The parameter of the prior Bernoulli distribution
        """
        return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))

    # We optimize the problem with the bisection method

    if (mode == "MAX"):
        p_max = 1 - 1e-10
        p_min = float(q)
    else:
        p_max = float(q)
        p_min = 1e-10
    q = min(max(q, 1e-10), 1 - 1e-10)
    for _ in range(nb_iter_max):
        p = (p_min + p_max) / 2.0
        if (kl(q, p) == epsilon or (p_max - p_min) / 2.0 < tol):
            return p

        if (mode == "MAX" and kl(q, p) > epsilon):
            p_max = p
        elif (mode == "MAX" and kl(q, p) < epsilon):
            p_min = p
        elif (mode == "MIN" and kl(q, p) > epsilon):
            p_min = p
        elif (mode == "MIN" and kl(q, p) < epsilon):
            p_max = p

    return p


def compute_bound(bound_info, meta_pred, pred, data_loader, n_sigma, m, r, delta, bnd_type, batch_size, a, b, inputs,
                  targets):
    """
    Sample compression bound of Marchand and someone else

    Parameters:
    -----------
    n_Z: uint
        Number of examples in the compression set
    n_sigma: uint
        Number of floating point values in the messages
    m: uint
        Number of examples in the training set
    r: uint
        Number of errors made by the classifier on the training set
    delta: float
        Confidence on the value of the bound is 1 - delta

    Returns:
    --------
    bound: float
        The bound value

    """
    n_Z = meta_pred.k
    n_sample = 2
    msg_type = bound_info
    best_bnd = 0
    n_grid = 11
    if msg_type == 'cnt':
        tot_acc, k = 0, 0
        # for inputs, targets in data_loader:
        #    inputs, targets = inputs.float(), targets.float()
        #    #    inputs_2, targets_2 = inputs_2.float(), targets_2.float()
        #    #if str(DEVICE) == 'gpu':
        #    #    inputs_1, targets_1, meta_pred = inputs_1.cuda(), targets_1.cuda(), meta_pred.cuda()
        meta_output = meta_pred(inputs, n_sample=n_sample)
        for sample in range(n_sample):
            outp = meta_output[[sample]]
            pred.update_weights(outp)
            output = pred.forward(inputs, outp, return_sign=True)
            acc = lin_loss(output[1], targets * 2 - 1, reduction=None)
            tot_acc += torch.sum(acc)
        tot_acc /= n_sample
        r = m - tot_acc
        KL = torch.mean(torch.sum(meta_pred.msg ** 2, dim=1))
        #print(KL)
        if bnd_type == 'kl':
            epsilon = (KL + np.log(2 * np.sqrt(m - n_Z) / zeta(n_Z) / delta)) / (m - n_Z)
            best_bnd = 1 - kl_inv(min((r / (m - n_Z)).item(), 1), epsilon.item(), 'MAX')
        elif bnd_type == 'linear':
            grid_start = -5
            for beta in np.logspace(grid_start, grid_start + n_grid):
                lambd = beta / m ** 0.5
                bound = 1 - ((r / (m - n_Z)) + lambd * (b - a) ** 2 / (8 * (m - n_Z)) +
                             (KL -
                              np.log(zeta(n_Z)) -
                              np.log(delta / n_grid)) / lambd
                             )
                if bound > best_bnd:
                    best_bnd = bound.cpu()
        elif bnd_type == 'hyperparam':
            grid_start = -5
            for beta in np.logspace(grid_start, grid_start + n_grid):
                C = beta / m ** 0.5
                bound = 1 - ((1 - math.exp(-C * (r / (m - n_Z)) -
                                           (KL -
                                            np.log(zeta(n_Z)) -
                                            np.log(delta / n_grid)) / (m - n_Z))) / (1 - math.e ** (-C)))
                if bound > best_bnd:
                    best_bnd = bound
        elif bnd_type == 'marchand':
            best_bnd = 0
    elif msg_type == 'dsc':
        p_sigma = 2 ** (-n_sigma)
        if bnd_type == 'kl':
            epsilon = (log_binomial_coefficient(m, n_Z) +
                       np.log(2 * np.sqrt(m - n_Z) / zeta(n_Z) / p_sigma / delta)) / (m - n_Z)
            best_bnd = 1 - kl_inv(min((r / (m - n_Z)).item(), 1), epsilon.item(), 'MAX').cpu()
        elif bnd_type == 'linear':
            grid_start = -5
            for beta in np.logspace(grid_start, grid_start + n_grid):
                lambd = beta / m ** 0.5
                bound = 1 - ((r / (m - n_Z)) + lambd * (b - a) ** 2 / 8 +
                             (log_binomial_coefficient(m, n_Z) -
                              np.log(p_sigma) -
                              np.log(zeta(n_Z)) -
                              np.log(delta / n_grid)) / (lambd * (m - n_Z))
                             )
                if bound > best_bnd:
                    best_bnd = bound.cpu()
        elif bnd_type == 'hyperparam':
            grid_start = -5
            for beta in np.logspace(grid_start, grid_start + n_grid):
                C = beta / m ** 0.5
                bound = 1 - ((1 - math.exp(-C * (r / (m - n_Z)) -
                                           (log_binomial_coefficient(m, n_Z) -
                                            np.log(p_sigma) -
                                            np.log(zeta(n_Z)) -
                                            np.log(delta / n_grid)) / (m - n_Z))) / (1 - math.e ** (-C))
                             )
                if bound > best_bnd:
                    best_bnd = bound
        elif bnd_type == 'marchand':
            best_bnd = 1 - sup_bin(int(r - n_Z), int(m - n_Z), delta * p_sigma * zeta(n_Z))
    return best_bnd
