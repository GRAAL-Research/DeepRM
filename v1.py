from scipy.integrate import tplquad, dblquad, quad
from scipy.special import gamma, betainc
from math import factorial
from time import time



def limit_x(x):
    return x


def limit_xy(x, y):
    return y / x

def limit_1_x(x):
    return 1 - x


def limit_mix(x, y):
    return (x + y - 1) / x


def limit_min(x, y):
    return max((x + y - 1) / x, 0)


def limit_max(x, y):
    return min(y / x, 1)


def binom(x, y):
    assert x >= y
    return gamma(x + 1) / (gamma(x - y + 1) * gamma(y + 1))


def Beta(x, y):
    return gamma(x) * gamma(y) / gamma(x + y)


for i in range(1,10):
    def integrand(alpha=1, beta=0, c=1, n=10, k_1=i, k_2_n=0, k_2_p=2):
        assert n >= k_1, f"n ({n}) must be superior or equal to k_1 ({k_1})."
        assert k_1 >= k_2_n, f"k_1 ({k_1}) must be superior or equal to k_2_m ({k_2_n})."
        assert (n - k_1) >= k_2_p, f"(n - k_1) ({(n - k_1)}) must be superior or equal to k_2_p ({k_2_p})."
        alpha_m = 1
        beta_m = c - alpha_m
        alpha_p = 0
        beta_p = c - alpha_p
        tot = 0
        for i in range(k_2_p + alpha_p):
            for j in range(n - k_1 - k_2_p + beta_p):
                sous_tot_1 = 0
                for h_1 in range(n + alpha + alpha_p + beta_p - i - j - 2, n + alpha + beta - i - j):
                    sous_tot_1 += binom(n + alpha + beta - i - j - 1, h_1)
                tot += Beta(i + 1, j + 1) * binom(k_2_p + alpha_p - 1, i) * binom(n - k_1 - k_2_p + beta_p - 1, j) * (-1) ** (n - k_1 + alpha_p + beta_p - i - j) * Beta(alpha_m + alpha_p + k_2_n + k_2_p - i - 1, beta_m + beta_p + n - k_2_n - k_2_p - j - 1) * \
                       Beta(n + alpha + alpha_p + beta_p - i - j - 2, beta - alpha_p - beta_p + 2) * sous_tot_1 * 0.5 ** (n + alpha + beta - i - j - 1)
                for l in range(i + 1, i + j + 2):
                    sous_tot_2 = 0
                    for h_2 in range(n + alpha + alpha_p + beta_p - 2 + l - i - j, n + alpha + beta + 1):
                        sous_tot_2 += binom(n + alpha + beta, h_2)
                    tot -= Beta(i + 1, j + 1) * binom(k_2_p + alpha_p - 1, i) * binom(n - k_1 - k_2_p + beta_p - 1, j) * (-1) ** (n - k_1 + alpha_p + beta_p - i - j) * Beta(alpha_m + alpha_p + k_2_n + k_2_p - i - 1, beta_m + beta_p + n - k_2_n - k_2_p - j - 1) * \
                           binom(i + j + 1, l) * Beta(n + alpha + alpha_p + beta_p - i - j - 2 + l, beta - alpha_p - beta_p + i + j + 3 - l) * sous_tot_2 * 0.5 ** (n + alpha + beta)

        return tot * (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) * \
                     (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) * \
                     (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p)))
    #one = tplquad(integrand, 0, 1, 0, 1, limit_min, limit_max)[0]
    #assert 1 - 1e-6 < one < 1 + 1e-6, f"{one}"
    t = time()
    tot = integrand()
    print(f"tot = {tot} for i = {i}.")