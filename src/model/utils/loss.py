import torch


def lin_loss(output, targets):
    """
    Computes the linear loss.
    Args:
        output (torch.tensor of size (batch_size, n_instances_per_dataset)): The output (0 or 1) of the predictor;
        targets (torch.tensor of size (batch_size, n_instances_per_dataset)): The labels (0 or 1);
    Return:
        Float, the total linear loss incurred.
    """
    if torch.sum(targets == 1) > 0:
        tot = torch.mean(((output[targets == 1] * targets[targets == 1]) + 1) / 4)
    else:
        tot = torch.tensor(0.5)
        print("This batch contains only examples from a single class.")
    if torch.sum(targets == -1) > 0:
        tot += torch.mean(((output[targets == -1] * targets[targets == -1]) + 1) / 4)
    else:
        tot += torch.tensor(0.5)
        print("This batch contains only examples from a single class.")
    return tot


def l1(x, c):
    """
    Computes the l1 loss, given inputs and a regularization parameter.
    Args:
        x (torch.tensor of size n_instances_per_dataset): Inputs
        c (float): Regularization parameter
    Return:
        Float, the l1 loss
    """
    return torch.mean(torch.abs(x)) * c


def l2(x, c):
    """
    Computes the l1 loss, given inputs and a regularization parameter.
    Args:
        x (torch.tensor of size n_instances_per_dataset): Inputs
        c (float): Regularization parameter
    Return:
        Float, the l2 loss
    """
    return torch.mean(x ** 2) * c
