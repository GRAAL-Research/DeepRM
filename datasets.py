import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def gen(m, d, dif='easy', vis=False, shuffle=True):
    """
    Generates a linearly separable 2-classes d-dimensional dataset. Each class
        is normaly distributed around a random mean with identity covariance matrix.
    Args:
        m (int): Number of examples in each class
        d (int): Input dimension of each dataset
        shuffle (bool): Whether the dataset is shuffled
    return:
        Tuple (X np.array
               y np.array)
    """
    if dif in ['easy', 'hard']:
        mu = np.random.rand(d) * 10 - 5
        X = np.random.multivariate_normal(mean=mu, cov=np.eye(d), size=2 * m)
        if dif == 'easy':
            X[:m] += np.sign(np.random.rand(d) - 0.5) * 5
        elif dif == 'hard':
            X[:m, np.random.randint(d)] += np.sign(np.random.randn()) * 5
        y = np.ones(2 * m)
        y[:m] -= 1
        for i in range(d):
            X[:, i] += np.random.rand() * 20 - 10
    if dif == 'moons':
        assert d == 2 # Moons must be of dimension 2
        scale = np.random.randint(3,8)
        rotation = np.random.randint(1,360)
        origin = np.random.randint(-10,10, 2)
        X = make_moons(n_samples=2 * m, shuffle=False, noise=0.05)[0] * scale + origin
        for i in range(2*m):
            new_coord = rotate((0,0), X[i], math.radians(rotation))
            X[i, 0], X[i, 1] = new_coord[0], new_coord[1]
        y = np.ones(2 * m)
        y[:m] -= 1
    if vis is True:
        plt.scatter(X[m:, 0], X[m:, 1], c='r')
        plt.scatter(X[:m, 0], X[:m, 1], c='b')
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.show()
    if shuffle:
        indx = np.arange(2 * m)
        np.random.shuffle(indx)
        X = X[indx]
        y = y[indx]
    return np.hstack((X,np.reshape(y*2-1, (2*m, 1)))), y


def data_gen(n, m, d, dif, shuffle=True, vis=False):
    """
    Generates a set of linearly separable 2-classes d-dimensional datasets.
    Args:
        n (int): Number of linearly separable datasets to create
        m (int): Number of examples per dataset
        d (int): Input dimension of each dataset
    return:
        Tuple of tuples (X 1-dim np.array of length 2m
                         y np.array of dims 2m x d)
    """
    assert d == 2 or vis == False
    m = int(m/2)
    X, y = gen(m, d, dif, vis, shuffle)
    data = [[X, y]]
    for i in range(n - 1):
        X, y = gen(m, d, dif, vis and i < 2, shuffle)
        data.append([X, y])
    return data
