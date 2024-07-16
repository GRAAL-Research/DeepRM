import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms


def load_mnist():
    """
    Generates a set of 90 MNIST binary sub-problems
    return:
        Numpy array of dims n x n_instances_per_dataset x n_features
    """
    # Loading the initial dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
    trainset = torch.hstack((trainset.data.reshape((60000, 28 * 28)), trainset.targets.reshape(60000, -1)))
    testset = torch.hstack((testset.data.reshape((10000, 28 * 28)), testset.targets.reshape(10000, -1)))
    dataset = torch.vstack((trainset, testset))

    # Creating 90 MNIST binary sub-problems
    data, k = np.zeros((90, 6313 * 2, 28 * 28 + 1)), 0
    indx = np.arange(2 * 6313)  # Randomize the position of the points in the dataset
    for i in range(10):
        for j in range(10):
            if i != j:
                x_1 = dataset[dataset[:, -1] == i, :-1]
                x_2 = dataset[dataset[:, -1] == j, :-1]
                x = torch.vstack((x_1[:6313], x_2[:6313]))
                y = torch.ones((6313 * 2, 1))
                y[:6313] -= 1
                np.random.shuffle(indx)
                data[k] = torch.hstack((x, y * 2 - 1))[indx]
                k += 1
    return data
