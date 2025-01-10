import torch


def linear_loss(output, targets):
    """
    Computes the linear loss.
    Args:
        output (torch.tensor of size (batch_size, n_instances_per_dataset)): The output (0 or 1) of the predictor;
        targets (torch.tensor of size (batch_size, n_instances_per_dataset)): The labels (0 or 1);
    Return:
        Float, the total linear loss incurred.
    """
    tot = []
    for i in range(len(targets)):
        if torch.sum(targets == 1) > 0:
            tot_i = torch.mean(((output[i, targets[i] == 1] * targets[i, targets[i] == 1]) + 1) / 4)
        else:
            tot_i = 0.5
            print("This batch contains only examples from a single class.")
        if torch.sum(targets == -1) > 0:
            tot_i += torch.mean(((output[i, targets[i] == -1] * targets[i, targets[i] == -1]) + 1) / 4)
        else:
            tot_i += 0.5
            print("This batch contains only examples from a single class.")
        tot.append(tot_i.item())
    return torch.tensor(tot)


def linear_loss_multi(output, targets):
    """
    Computes the linear loss.
    Args:
        output (torch.tensor of size (batch_size, n_instances_per_dataset)): The output (0 or 1) of the predictor;
        targets (torch.tensor of size (batch_size, n_instances_per_dataset)): The labels (0 or 1);
    Return:
        Float, the total linear loss incurred.
    """
    return torch.sum(output * targets, dim=-1)


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
