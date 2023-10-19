import math
import numpy as np
from scipy.stats import norm

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

def zeta(x):
    """
    Mario's function...
    """
    return (6 / np.pi**2) * (x + 1)**-2

def prior(sigma):
    """
    Prior on the value of messages (this is an arbitrary choice)

    """
    # Length prior: we make an arbitrary choice that decays with message length (favor smaller messages)
    p_len = zeta(len(sigma))

    # Content prior: we use a simple isotropic standard normal prior (favors small weights)
    p_content = np.prod(norm.pdf(sigma))  # Product of likelihood of each dimension

    return p_len * p_content

def compute_bound(bound_type, n_Z, n_sigma, m, r, c_2, d, delta):
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
    # For now, we craft an arbitrary message by sampling n_sigma values from a N01 (matches the prior)
    if bound_type == 'Alex':
        sigma = np.random.randn(n_sigma) * 2
        p_sigma = prior(sigma)
        bound = np.exp(
                    -1 / (m - n_Z - r)*
                     (
                        log_binomial_coefficient(m, n_Z) +
                        log_binomial_coefficient(m - n_Z, r) +
                        np.log(1 / p_sigma) +
                        np.log(1 / (zeta(n_Z) * zeta(r) * delta))
                     )
                )
    elif bound_type == 'Mathieu':
        bound = 1 - (r / m + math.sqrt(c_2 * (d + log_binomial_coefficient(m, n_Z) - math.log(zeta(n_Z) * delta)) / m))
    return bound