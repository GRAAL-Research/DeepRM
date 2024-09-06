from scipy.integrate import tplquad
from scipy.special import gamma
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


for i in range(0,10):
    def integrand(z, y, x, alpha=1, beta=0, c=1, n=10, k_1=i, k_2_n=0, k_2_p=1):
        assert n >= k_1, f"n ({n}) must be superior or equal to k_1 ({k_1})."
        assert k_1 >= k_2_n, f"k_1 ({k_1}) must be superior or equal to k_2_m ({k_2_n})."
        assert (n - k_1) >= k_2_p, f"(n - k_1) ({(n - k_1)}) must be superior or equal to k_2_p ({k_2_p})."
        alpha_m = 1
        beta_m = c - alpha_m
        alpha_p = 0
        beta_p = c - alpha_p
        ## P( R(h_1) <= R(h_2) ) ##
        return (x ** (alpha + k_1 - 1) * (1 - x) ** (beta + (n - k_1) - 1) *
                z ** (alpha_m + k_2_n - 1) * (1 - z) ** (beta_m + (k_1 - k_2_n) - 1) *
                ((y - z * x) / (1 - x)) ** (alpha_p + k_2_p - 1) *
                (1 - (y - z * x) / (1 - x)) ** (beta_p + ((n - k_1) - k_2_p) - 1) *
                (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) *
                (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) *
                (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p))))

        ## E( R(h_1) - R(h_2) ) ##
        #return (x ** (alpha + k_1 - 0) * (1 - x) ** (beta + (n - k_1) - 1) *
        #        z ** (alpha_m + k_2_n - 1) * (1 - z) ** (beta_m + (k_1 - k_2_n) - 1) *
        #        ((y - z * x) / (1 - x)) ** (alpha_p + k_2_p - 1) *
        #        (1 - (y - z * x) / (1 - x)) ** (beta_p + ((n - k_1) - k_2_p) - 1) *
        #        (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) *
        #        (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) *
        #        (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p)))) - \
        #    (x ** (alpha + k_1 - 1) * (1 - x) ** (beta + (n - k_1) - 1) * y *
        #     z ** (alpha_m + k_2_n - 1) * (1 - z) ** (beta_m + (k_1 - k_2_n) - 1) *
        #     ((y - z * x) / (1 - x)) ** (alpha_p + k_2_p - 1) *
        #     (1 - (y - z * x) / (1 - x)) ** (beta_p + ((n - k_1) - k_2_p) - 1) *
        #     (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) *
        #     (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) *
        #     (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p))))

        ## P( R(h_1) <= R(h_2) ) ##
        #tot = 0
        #for i in range(k_2_p + alpha_p):
        #    for j in range(n - k_1 - k_2_p + beta_p):
        #        tot += binom(k_2_p + alpha_p - 1, i) * binom(n - k_1 - k_2_p + beta_p - 1, j) * (-1) ** (n - k_1 + alpha_p + beta_p - i - j) * \
        #                     x ** (n + alpha + alpha_p + beta_p - i - j - 3) * (1-x) ** (beta - alpha_p - beta_p + 1) * \
        #                     y ** (i) * (1 - y) ** (j) * \
        #                     z ** (alpha_m + alpha_p + k_2_n + k_2_p - i - 2) * (1 - z) ** (beta_m + beta_p + n - k_2_n - k_2_p - j - 2)
        #return tot * (gamma(alpha + beta + n + 1) / gamma(alpha + k_1) / gamma(beta + (n - k_1) + 1)) * \
        #             (gamma(c + k_1) / gamma(alpha_m + k_2_n) / gamma(beta_m + (k_1 - k_2_n))) * \
        #             (gamma(c + (n - k_1)) / gamma(alpha_p + k_2_p) / gamma(beta_p + ((n - k_1) - k_2_p)))
    #one = tplquad(integrand, 0, 1, 0, 1, limit_min, limit_max)[0]
    #assert 1 - 1e-6 < one < 1 + 1e-6, f"{one}"
    t = time()
    pro  = tplquad(integrand, 0, 0.5, 0, limit_x, 0, limit_xy)[0]
    pro += tplquad(integrand, 0.5, 1, 0, limit_1_x, 0, limit_xy)[0]
    pro += tplquad(integrand, 0.5, 1, limit_1_x, limit_x, limit_mix, limit_xy)[0]

    #tot = 0
    #tot += tplquad(integrand, 0, 0.5, 0, limit_x, 0, limit_xy)[0]
    #tot += tplquad(integrand, 0.5, 1, 0, limit_1_x, 0, limit_xy)[0]
    #tot += tplquad(integrand, 0.5, 1, limit_1_x, limit_x, limit_mix, limit_xy)[0]

    #tot += tplquad(integrand, 0, 0.5, limit_x, limit_1_x, 0, 1)[0]
    #tot += tplquad(integrand, 0, 0.5, limit_1_x, 1, limit_mix, 1)[0]
    #tot += tplquad(integrand, 0.5, 1, limit_x, 1, limit_mix, 1)[0]

    #tot += tplquad(integrand, 0, 0.5, limit_x, 1, 0, 1)[0]
    #tot -= tplquad(integrand, 0, 0.5, limit_1_x, 1, 0, limit_mix)[0]
    #tot += tplquad(integrand, 0.5, 1, limit_x, 1, limit_mix, 1)[0]
    print(f"tot = {pro} for i = {i}.")
    #print(f"P( R(h_1) >= R(h_2) ) >= {round(pro * 100, 6)}% for i = {i}.")
    #print(f"Took {round(time() - t, 3)} seconds.")