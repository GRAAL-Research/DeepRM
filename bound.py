import math
import torch
from utils import *


def log_stirling_approximation(n):
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
    return (log_stirling_approximation(n) -
            log_stirling_approximation(k) -
            log_stirling_approximation(n - k))


def log_prob_bin(k, n, r):
    """
    Logarithm of P(x = k), if X ~ Bin(n, r)
    """
    return log_binomial_coefficient(n, k) + k * math.log(max(r, 1e-10)) + (n - k) * math.log(max(1 - r, 1e-10))


def bin_cum(k, n, r):
    """
    Logarithm of P(x <= k), if X ~ Bin(n, r)
    """
    prob_cum = 0
    for i in range(k + 1):
        prob_cum += math.exp(log_prob_bin(i, n, r))
    return prob_cum


def sup_bin(k, m, delta):
    """
    Estimation of sup(r : P(x <= k) >= delta), if X ~ Bin(m, r)
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(10):
        pro = bin_cum(k, m, gamma)
        if pro >= delta:
            gamma_inf = gamma
        else:
            gamma_sup = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma


def inf_bin(k, m, delta):
    """
    Estimation of inf(r : P(x <= k) >= 1 - delta), if X ~ Bin(m, r)
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(10):
        pro = bin_cum(k, m, gamma)
        if 1 - pro >= delta:
            gamma_sup = gamma
        else:
            gamma_inf = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma


def zeta(x):
    """
    Mario's function...
    """
    return (6 / np.pi ** 2) * (x + 1) ** -2


def kl_bern(q, p):
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
    mode: str
        Type of optimization (see function description) (choices: 'MIN', 'MAX')
    tol: float, optional
        The precision tolerance of the solution
    nb_iter_max: int, optional
        The maximum number of iterations
    """
    assert mode == "MIN" or mode == "MAX"
    assert 0 <= q <= 1, f"q is out of bounds: must be within [0,1], has value {q}."
    assert isinstance(epsilon, float) and epsilon > 0.0

    # We optimize the problem with the bisection method
    p = 0
    if mode == "MAX":
        p_max = 1 - 1e-10
        p_min = float(q)
    else:
        p_max = float(q)
        p_min = 1e-10
    q = min(max(q, 1e-10), 1 - 1e-10)
    for _ in range(nb_iter_max):
        p = (p_min + p_max) / 2.0
        if kl_bern(q, p) == epsilon or (p_max - p_min) / 2.0 < tol:
            return p
        if mode == "MAX" and kl_bern(q, p) > epsilon:
            p_max = p
        elif mode == "MAX" and kl_bern(q, p) < epsilon:
            p_min = p
        elif mode == "MIN" and kl_bern(q, p) > epsilon:
            p_min = p
        elif mode == "MIN" and kl_bern(q, p) < epsilon:
            p_max = p
    return p


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
                pred.update_weights(outp)
                output = pred.forward(inputs, outp, return_sign=True)
                acc = lin_loss(output[1], targets * 2 - 1, reduction=False)
                tot_acc += torch.sum(acc)
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
