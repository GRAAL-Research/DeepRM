import math

from src.utils import *


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


