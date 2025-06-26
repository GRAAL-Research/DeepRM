import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.mlp import MLP
from src.model.utils.loss import linear_loss, linear_loss_multi
from src.training.compute_loss import compute_loss
from src.training.factory.optimizer import create_optimizer


def launch_prior_training(config: dict, prior: MLP, train_loader: DataLoader, test_loader: DataLoader,
                          criterion: nn.Module) -> MLP:
    """
    The prior corresponds to the single model capable of achieving the best performances over the meta-train tasks.
    """
    optimizer = create_optimizer(config, prior)
    indx_vec = np.arange(config["n_instances_per_dataset"])
    if config["target_size"] == 1:
        output_activation = torch.nn.Sigmoid()
    else:
        output_activation = torch.nn.Softmax(dim=2)
    print("*** Prior training ***")
    for epoch_idx in range(config["max_prior_epoch"]):
        prior.train()
        for instances in train_loader:
            if config["device"] == "gpu":
                instances = instances.cuda()
            instances = instances.float()
            with (torch.enable_grad()):
                targets = (instances[:, :, -config["target_size"]:] + 1) / 2
                feature = instances[:, :, :-config["target_size"]]
                optimizer.zero_grad()
                output = prior(feature)
                loss = compute_loss(config, criterion, output_activation(output), targets, None)
                loss.backward()
                optimizer.step()

        # The prior model now is in evaluation mode; the batch norm layers are not impacted by the test data.
        prior.eval()
        tot_acc = []
        for instances in test_loader:
            np.random.shuffle(indx_vec)
            targets = (instances[:, indx_vec, -config["target_size"]:] + 1) / 2
            instances = instances[:, indx_vec, :-config["target_size"]]

            instances = instances.float()
            targets = targets.float()
            if config["device"] == "gpu":
                instances = instances.cuda()
                targets = targets.cuda()
            output = prior(instances)

            if config["criterion"] == "bce_loss":
                output = torch.sign(output)
            elif config["criterion"] == "ce_loss":
                output = torch.nn.functional.one_hot(torch.argmax(output, dim=-1),
                                                     num_classes=output.shape[-1])
            if config["target_size"] == 1:
                acc = torch.mean(linear_loss(output, targets * 2 - 1), dim=-1)
            else:
                acc = torch.mean(linear_loss_multi(output, targets), dim=-1)
            tot_acc.append(torch.mean(acc).item())
        print(f"Epoch #{epoch_idx + 1}: {round(np.mean(tot_acc), 4)}% test accuracy")
    print("\n*** Posterior training ***")
    return prior
