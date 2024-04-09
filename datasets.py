import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import torch
import torchvision
import torchvision.transforms as transforms
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

def gen(m, d, dif='easy', vis=False, shuffle=True, i=0):
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
    if dif == 'both':
        rand = np.random.randint(2)
        dif = rand * 'easy' + (1-rand) * 'moons'
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
        X_1 = make_moons(n_samples=2 * m, shuffle=False, noise=0.08, random_state=i)[0] * scale + origin
        X_2 = make_moons(n_samples=2 * 3, shuffle=False, noise=0.08, random_state=i)[0] * scale + origin
        for i in range(2 * m):
            new_coord = rotate((0, 0), X_1[i], math.radians(rotation))
            X_1[i, 0], X_1[i, 1] = new_coord[0], new_coord[1]
        for i in range(2 * 3):
            new_coord = rotate((0, 0), X_2[i], math.radians(rotation))
            X_2[i, 0], X_2[i, 1] = new_coord[0], new_coord[1]
        y_1 = np.ones(2 * m)
        y_2 = np.ones(2 * 3)
        y_1[:m] -= 1
        y_2[:3] -= 1
    if vis > 0:
        plt.scatter(X_1[m:, 0], X_1[m:, 1], c='r')
        plt.scatter(X_1[:m, 0], X_1[:m, 1], c='b')
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.show()
        plt.scatter(X_2[3:, 0], X_2[3:, 1], c='r')
        plt.scatter(X_2[:3, 0], X_2[:3, 1], c='b')
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.show()
    if shuffle:
        indx = np.arange(2 * m)
        np.random.shuffle(indx)
        X_1 = X_1[indx]
        y_1 = y_1[indx]
        indx = np.arange(2 * 3)
        np.random.shuffle(indx)
        X_2 = X_2[indx]
        y_2 = y_2[indx]
    return np.hstack((X_1,np.reshape(y_1*2-1, (2*m, 1)))), y_1, np.hstack((X_2,np.reshape(y_2*2-1, (2*3, 1)))), y_2


def data_gen(n, m, d, dif, shuffle=True, vis=False):
    """
    Generates a set of linearly separable 2-classes d-dimensional datasets.
    Args:
        n (int): Number of linearly separable datasets to create
        m (int): Number of examples per dataset
        d (int): Input dimension of each dataset
    return:
        Tuple of tuples (X np.array of dims 2m x d
                         y 1-dim np.array of length 2m)
    """
    assert d == 2 or vis == False
    m = int(m/2)
    X_1, y_1, X_2, y_2 = gen(m, d, dif, vis, shuffle)
    data_1 = [[X_1, y_1]]
    data_2 = [[X_2, y_2]]
    for i in range(n - 1):
        vis -= 1
        X_1, y_1, X_2, y_2 = gen(m, d, dif, vis, shuffle, i)
        data_1.append([X_1, y_1])
        data_2.append([X_2, y_2])
    return data_1, data_2

def load_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./datasets', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./datasets', train=False,
                                           download=True, transform=transform)
    trainset = torch.hstack((trainset.data.reshape((60000, 28 * 28)), trainset.targets.reshape(60000, -1)))
    testset = torch.hstack((testset.data.reshape((10000, 28 * 28)), testset.targets.reshape(10000, -1)))
    dataset = torch.vstack((trainset, testset))
    data = []
    for i in range(10):
        for j in range(10):
            if i != j:
                X_1 = dataset[dataset[:, -1] == i, :-1]
                X_2 = dataset[dataset[:, -1] == j, :-1]
                X = torch.vstack((X_1[:6313], X_2[:6313]))
                y = torch.zeros((6313*2))
                y[6313:] += 1
                data.append([torch.hstack((X, torch.reshape(y, (-1,1)))), y])
    return data # Tuple de tuple (2m x d, 2m)