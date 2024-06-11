import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import torch
import torchvision
import torchvision.transforms as transforms
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.
    Args:
        origin ([float, float]): x and y coordinate of the origin around which to rotate;
        point ([float, float]]): x and y coordinate of the point to rotate;
        angle (float): angle of rotation, in radians.
    return:
        [float, float], coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def gen(dataset, m, d, shuffle=True):
    """
    Generates a linearly separable 2-classes d-dimensional dataset. Each class
        is normaly distributed around a random mean with identity covariance matrix.
    Args:
        dataset (str): Name of the dataset to generate;
        m (int): Number of examples in each class
        d (int): Input dimension of each dataset
        shuffle (bool): Whether the dataset is shuffled
    return:
        Tuple
    """
    X = []
    if dataset == 'both':
        rand = np.random.randint(2)  # Randomly chooses between 'easy' and 'moons' dataset
        dataset = rand * 'easy' + (1-rand) * 'moons'
    if dataset == 'easy':     # Corresponds to Gaussian blobs
        mu = np.random.rand(d) * 10 - 5  # Random center for the Gaussian blobs
        X = np.random.multivariate_normal(mean=mu, cov=np.eye(d), size=2 * m)
        X[:m] += np.sign(np.random.rand(d) - 0.5) * 5   # Making a class distinct from the other
    if dataset == 'moons':
        assert d == 2  # Moons must be of dimension 2
        # Three parameters constitute a dataset: the origin of the moons, their rotation around the origin and its scale
        scale = np.random.randint(3,8)
        rotation = np.random.randint(1,360)
        origin = np.random.randint(-10,10, 2)
        X = make_moons(n_samples=2 * m, shuffle=False, noise=0.08)[0] * scale + origin
        for i in range(2 * m):
            X[i, 0], X[i, 1] = rotate((0, 0), X[i], math.radians(rotation))    # We rotate, one by one, the points
    y = np.ones((2 * m, 1))
    y[:m] -= 1
    if shuffle:
        indx = np.arange(2 * m)  # Randomize the position of the points in the dataset
        np.random.shuffle(indx)
        X, y = X[indx], y[indx]
    return [np.hstack((X, y)), np.squeeze(y)]


def data_gen(dataset, n, m, d, shuffle=True):
    """
    Generates a set of linearly separable 2-classes d-dimensional datasets.
    Args:
        dataset (str): Name of the dataset to generate;
        n (int): Number of linearly separable datasets to create;
        m (int): Number of examples per dataset;
        d (int): Input dimension of each dataset;
        shuffle (bool): whether to shuffle the points in each generated dataset
    return:
        Tuple of tuples (np.array of dims 2m x d,
                         np.array of length 2m)
    """
    assert d == 2   # Only generates 2D datasets
    m, data = int(m/2), []
    for i in range(n):
        data.append(gen(dataset, m, d, shuffle))
    return data


def load_mnist():
    """
    Generates a set of 90 MNIST binary sub-problems
    return:
        List of [X np.array of dims 2m x d,
                 y 1-dim np.array of length 2m]
    """
    # Loading the initial dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    trainset = torch.hstack((trainset.data.reshape((60000, 28 * 28)), trainset.targets.reshape(60000, -1)))
    testset = torch.hstack((testset.data.reshape((10000, 28 * 28)), testset.targets.reshape(10000, -1)))
    dataset = torch.vstack((trainset, testset))

    # Creating 90 MNIST binary sub-problems
    data = []
    for i in range(10):
        for j in range(10):
            if i != j:
                X_1 = dataset[dataset[:, -1] == i, :-1]
                X_2 = dataset[dataset[:, -1] == j, :-1]
                X = torch.vstack((X_1[:6313], X_2[:6313]))
                y = torch.ones((6313*2))
                y[6313:] -= 1
                data.append([torch.hstack((X, torch.reshape(y, (-1,1)))), y])
    return data