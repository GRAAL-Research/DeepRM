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
    def integrand(alpha=1, beta=0, n=10, k_1=i, k_2_n=0, k_2_p=2):
        assert n >= k_1, f"n ({n}) must be superior or equal to k_1 ({k_1})."
        assert k_1 >= k_2_n, f"k_1 ({k_1}) must be superior or equal to k_2_m ({k_2_n})."
        assert (n - k_1) >= k_2_p, f"(n - k_1) ({(n - k_1)}) must be superior or equal to k_2_p ({k_2_p})."
        tot = 0
        for i in range(k_2_p):
            for j in range(n - k_1 - k_2_p + 1):
                tot += Beta(i + 1, j + 1) * binom(k_2_p - 1, i) * binom(n - k_1 - k_2_p, j) * (-1) ** (n - k_1 - i - j + 1) * Beta(k_2_n + k_2_p - i, n - k_2_n - k_2_p - j) * \
                       (n - i - j) ** (-1) * 0.5 ** (n - i - j)
                for l in range(i + 1, i + j + 2):
                    sous_tot_2 = 0
                    for h_2 in range(n + l - i - j, n + 2):
                        sous_tot_2 += binom(n + alpha + beta, h_2)
                    print(sous_tot_2)
                    tot -= Beta(i + 1, j + 1) * binom(k_2_p - 1, i) * binom(n - k_1 - k_2_p, j) * (-1) ** (n - k_1 - i - j + 1) * Beta(k_2_n + k_2_p - i, n - k_2_n - k_2_p - j) * \
                           binom(i + j + 1, l) * Beta(n - i - j + l, i + j + 2 - l) * sous_tot_2 * 0.5 ** (n + alpha + beta)

        return tot * (gamma(n + 2) / gamma(k_1 + 1) / gamma((n - k_1) + 1)) * \
                     (gamma(k_1 + 1) / gamma(k_2_n + 1) / gamma(k_1 - k_2_n)) * \
                     (gamma(n - k_1 + 1) / gamma(k_2_p) / gamma(n - k_1 - k_2_p + 1))
    #one = tplquad(integrand, 0, 1, 0, 1, limit_min, limit_max)[0]
    #assert 1 - 1e-6 < one < 1 + 1e-6, f"{one}"
    t = time()
    tot = integrand()
    print(f"tot = {tot} for i = {i}.")