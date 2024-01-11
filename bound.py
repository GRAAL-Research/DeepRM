import math
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

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

def eta(x,m):
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

def a_star(m,n_Z,r,a):
    return (1-
            np.exp(1/(m-n_Z-r) * (r * np.log(r/(m-n_Z)) + (m - n_Z - r) * np.log(1 - r/(m-n_Z)) - r * np.log(r/(m-n_Z) + a))) -
            r/(m-n_Z))
def bnd(m,n_Z,r,p_sigma,delta,a):
    return 1-np.exp(
        -1 / (m - n_Z - r) *
        (r * np.log(r/(m-n_Z)+a)+
         log_binomial_coefficient(m, n_Z) +
         log_binomial_coefficient(m - n_Z, r) +
         np.log(1 / p_sigma) +
         np.log(1 / (zeta(n_Z) * zeta(r) * delta))
         ))

def compute_bound(bound_type, n_Z, n_sigma, m, r, c_2, d, acc, delta):
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
    sigma = np.random.randn(n_sigma) * 2
    p_sigma = prior(sigma)
    if bound_type == 'Alex':
        bound = np.exp(
                    -1 / (m - n_Z - r)*
                     (  log_binomial_coefficient(m, n_Z) +
                        log_binomial_coefficient(m - n_Z, r) +
                        np.log(1 / p_sigma) +
                        np.log(1 / zeta(n_Z)) +
                        np.log(1 / zeta(r)) +
                        np.log(1 / delta)
                     )
                )
    if bound_type == 'Ben2':
        ini_bnd = np.exp(
                    -1 / (m - n_Z - r)*
                     (  log_binomial_coefficient(m, n_Z) +
                        log_binomial_coefficient(m - n_Z, r) +
                        np.log(1 / p_sigma) +
                        np.log(1 / (zeta(n_Z) * zeta(r) * delta))
                     )
                )
        maxx, best_c = 0, 0
        bnd = []
        for c in range(1,int((m - n_Z - r).item())):
            bound = np.exp(
                -1 / (m - n_Z - r - c) *
                (log_binomial_coefficient(m, n_Z) +
                 log_binomial_coefficient(m - n_Z, r) +
                 r * np.log(r / (r + c)) +
                 c * np.log(c / (r + c)) +
                 np.log(1 / p_sigma) +
                 np.log(1 / (zeta(n_Z) * zeta(r) * delta))
                 )
            )
            bnd.append(np.copy(bound))
            if np.copy(bound) > maxx:
                maxx, best_c = np.copy(bound), c
        print()
        print(f"Alex's bound: {ini_bnd:.4f}")
        print(f"Ben's bound: {maxx:.4f}")
        print(f"Gain: {maxx-np.array(ini_bnd):.4f}")
        print()
        bound = ini_bnd #maxx
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        plt.plot(bnd)
        plt.hlines(ini_bnd, 1, int((m - n_Z - r).item()), linestyles='solid', colors='orange')
        plt.hlines((m - n_Z - r) / (m - n_Z), 1, int((m - n_Z - r).item()), linestyles='solid', colors='green')
        plt.hlines(acc[0].item(), 1, int((m - n_Z - r).item()), linestyles='dashed', colors='green')
        plt.hlines(acc[1].item(), 1, int((m - n_Z - r).item()), linestyles='dotted', colors='green')
        plt.vlines(r/(1-maxx)-r, 0, 0.8, colors='red')
        plt.legend(['Ben bound', 'Marchand Sokolova bound', 'Training acc', 'Validation acc', 'Test acc', 'Best'])
        plt.xlabel('c')
        plt.ylabel('Accuracy / bound value')
        plt.savefig("figures/bnd_Ben.png")
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
    elif bound_type == 'Mathieu':
        bound = 1 - (r / m + math.sqrt(c_2 * (d + log_binomial_coefficient(m, n_Z) - math.log(zeta(n_Z) * delta)) / m))
    return bound
