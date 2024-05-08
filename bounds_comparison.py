from matplotlib import pyplot as plt
import torch
from utils import *
from bound import *

def compute_bound(msg_type, bnd_type, m, r, n_Z, p_sigma, delta, a, b):
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
    best_bnd = 0
    n_grid = 11
    if msg_type == 'cnt':
        KL = p_sigma
        if bnd_type == 'kl':
            epsilon = (KL + np.log(2 * np.sqrt(m - n_Z) / zeta(n_Z) / delta)) / (m - n_Z)
            best_bnd = 1 - kl_inv((r / (m - n_Z)), epsilon, 'MAX')
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
                    best_bnd = bound
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
        if bnd_type == 'kl':
            epsilon = (log_binomial_coefficient(m, n_Z) +
                       np.log(2 * np.sqrt(m - n_Z) / zeta(n_Z) / p_sigma / delta)) / (m - n_Z)
            best_bnd = 1 - kl_inv((r / (m - n_Z)), epsilon, 'MAX')
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
                    best_bnd = bound
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
            #best_bnd = 1 - sup_bin(int(r - n_Z), int(m - N_Z), delta * p_sigma * zeta(n_Z))
            best_bnd = math.exp((-1 / (m - r - n_Z)) * (
                             log_binomial_coefficient(m - n_Z, r) -
                             np.log(p_sigma) -
                             np.log(zeta(n_Z)) -
                             np.log(delta)
                             )
                           )
    return best_bnd

"""
-dsc: 
    -n: Number of examples in the training set
    -r: Percentage of errors made by the classifier on the training set
    -i: Number of examples in the compression set
    -p_sigma: probability of obtaining a given message (2 ** -n_sigma)
    -delta: confidence level
    
-cnt:
    -n: Number of examples in the training set
    -r: Number of errors made by the classifier on the training set
    -i: Number of examples in the compression set
    -KL: KL divergence between prior and posterior
    -delta: confidence level
"""

#n = [100, 1000, 10000]
#r_frac = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
#i_frac = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
#p_sigma = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
#delta = 0.01

ns = [10000, 1000, 100]
r_fracs = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
i_frac = 0.005
p_sigma = 1e-1
delta = 0.01
msg_type = 'dsc'

lss = ['--', '-.', ':']
for it in range(len(ns)):
    kl = []
    lin = []
    hyp = []
    mrch = []
    m = ns[it]
    n_Z = int(m * i_frac)
    for r_frac in r_fracs:
        r = int((m - n_Z) * r_frac)
        print(m, r, n_Z)
        mrch.append(compute_bound(msg_type, 'marchand', m, r, n_Z, p_sigma, delta, 0, 1))
        kl.append(compute_bound(msg_type, 'kl', m, r, n_Z, p_sigma, delta, 0, 1))
        hyp.append(compute_bound(msg_type, 'hyperparam', m, r, n_Z, p_sigma, delta, 0, 1))
        lin.append(compute_bound(msg_type, 'linear', m, r, n_Z, p_sigma, delta, 0, 1))
    plt.plot(r_fracs, mrch, c='blue', ls=lss[it])
    plt.plot(r_fracs, kl, c='red', ls=lss[it])
    plt.plot(r_fracs, hyp, c='green', ls=lss[it])
    plt.plot(r_fracs, lin, c='black', ls=lss[it])
    plt.plot(r_fracs, 1-np.array(r_fracs), c='yellow')
plt.title("Accuracy bound as a function of the fraction of error made")
plt.legend([f"March (n={ns[0]})", f"KL (n={ns[0]})", f"Hyp (n={ns[0]})", f"Lin (n={ns[0]})", f"Train acc.",
            f"March (n={ns[1]})", f"KL (n={ns[1]})", f"Hyp (n={ns[1]})", f"Lin (n={ns[1]})", f"Train acc.",
            f"March (n={ns[2]})", f"KL (n={ns[2]})", f"Hyp (n={ns[2]})", f"Lin (n={ns[2]})"])
plt.xlabel('Proportion of error on the training set')
plt.ylabel('Bound on the 0-1 accuracy')
plt.show()