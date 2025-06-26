import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.training.compute_loss import compute_loss


def launch_epoch_training(config: dict, meta_predictor: SimpleMetaNet, predictor: Predictor, train_loader: DataLoader,
                          criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Predictor:
    """
    A standard meta-learning loop.
    """
    meta_predictor.train()
    batch_size = config["batch_size"] if config["batch_size"] > 0 else config["n_instances_per_dataset"] + 1
    n_instances_per_class_per_dataset = batch_size // 2
    with (torch.enable_grad()):
        # For all train task...
        for meta_instances in train_loader:
            meta_instances = meta_instances[0]
            # ... And for all batch in every task.
            for i_batch in range((len(meta_instances[0]) // batch_size) + 1):
                instances = meta_instances[:, i_batch * batch_size:(i_batch + 1) * batch_size]
                targets = (instances[:, :, -config["target_size"]:] + 1) / 2
                instances = instances.float()
                targets = targets.float()
                if config["device"] == "gpu":
                    instances = instances.cuda()
                    targets = targets.cuda()
                    meta_predictor = meta_predictor.cuda()

                optimizer.zero_grad()
                meta_output = meta_predictor.forward(instances[:, :n_instances_per_class_per_dataset])
                predictor.set_params(meta_output)
                output, _ = predictor.forward(instances[:, n_instances_per_class_per_dataset:])
                loss = compute_loss(config, criterion, output, targets[:, n_instances_per_class_per_dataset:],
                                    meta_predictor)
                loss.backward()
                optimizer.step()

    return predictor
