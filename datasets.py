import numpy as np
import math
from matplotlib import pyplot as plt


def gen_2d(m, vis=False):
    """
    Generates a linearly separable 2-classes 2-dimensional dataset. Each class
        is normaly distributed around a random mean with identity covariance matrix.
    Args:
        m (int): Number of examples in each class
        vis (bool): Whether to display the created dataset
    return:
        Tuple (X np.array
               y np.array)
    """
    ang = np.random.rand() * math.pi * 2
    mu = np.array([math.sin(ang), math.cos(ang)])
    X = np.random.multivariate_normal(mean=mu * 5, cov=np.eye(2), size=2 * m)
    X[:m] *= -1
    y = np.ones(2 * m)
    y[:m] -= 1
    X[:, 0] += np.random.rand() * 20 - 10
    X[:, 1] += np.random.rand() * 20 - 10
    if vis is True:
        plt.scatter(X[m:, 0], X[m:, 1], c='r')
        plt.scatter(X[:m, 0], X[:m, 1], c='b')
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.show()
    return X, y


def gen_d(m, d):
    """
    Generates a linearly separable 2-classes d-dimensional dataset. Each class
        is normaly distributed around a random mean with identity covariance matrix.
    Args:
        m (int): Number of examples in each class
        d (int): Input dimension of each dataset
        vis (bool): Whether to display the created dataset
    return:
        Tuple (X np.array
               y np.array)
    """
    mu = np.random.rand(d) * 10 + 5
    X = np.random.multivariate_normal(mean=mu, cov=np.eye(d), size=2 * m)
    X[:m] *= -1
    y = np.ones(2 * m)
    y[:m] -= 1
    for i in range(d):
        X[:, i] += np.random.rand() * 20 - 10
    return X, y


def data_gen(n, m, d):
    """
    Generates a set of linearly separable 2-classes d-dimensional datasets.
    Args:
        n (int): Number of linearly separable datasets to create
        m (int): Number of examples in each class, per dataset
        d (int): Input dimension of each dataset
    return:
        Tuple of tuples (X 1-dim np.array of length 2m
                         y np.array of dims 2m x d)
    """
    gen = gen_2d if d == 2 else gen_d
    X, y = gen(m, d)
    data = [[X, y]]
    for i in range(n - 1):
        X, y = gen(m, d)
        data.append([X, y])
    return data
