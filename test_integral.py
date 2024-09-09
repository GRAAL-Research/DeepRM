from scipy.integrate import tplquad, dblquad
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
    def integrand(y, x, alpha=1, beta=0, c=1, n=10, k_1=i, k_2_n=0, k_2_p=2):
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
                for l in range(alpha_m + alpha_p + k_2_n + k_2_p - i - 1, n + alpha_m + alpha_p + beta_m + beta_p - i - j - 2):
                    for h in range(0, l + 1):
                        tot += binom(k_2_p + alpha_p - 1, i) * binom(n - k_1 - k_2_p + beta_p - 1, j) * (-1) ** (n - k_1 + alpha_p + beta_p - i - j) * \
                                     x ** (n + alpha + alpha_p + beta_p - i - j - 3) * (1-x) ** (beta - alpha_p - beta_p + 1) * \
                                     y ** (i) * (1 - y) ** (j) * \
                               Beta(alpha_m + alpha_p + k_2_n + k_2_p - i - 1, beta_m + beta_p + n - k_2_n - k_2_p - j - 1) * \
                               binom(l, h) * (-1) ** h * binom(n + alpha_m + alpha_p + beta_m + beta_p - i - j - 3, l) * \
                               ((1 - y) / x) ** (n + alpha_m + alpha_p + beta_m + beta_p - i - j - 3 - l + h)
        return tot * (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) * \
                     (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) * \
                     (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p)))
    #one = tplquad(integrand, 0, 1, 0, 1, limit_min, limit_max)[0]
    #assert 1 - 1e-6 < one < 1 + 1e-6, f"{one}"
    t = time()
    tot = 0
    tot += dblquad(integrand, 0, 0.5, limit_1_x, 1)[0]
    print(f"tot = {tot} for i = {i}.")
    #print(f"P( R(h_1) >= R(h_2) ) >= {round(pro * 100, 6)}% for i = {i}.")
    #print(f"Took {round(time() - t, 3)} seconds.")