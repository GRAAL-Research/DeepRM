import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.bound.compute_bound import compute_bounds
from src.model.predictor.predictor import Predictor
from src.model.simple_meta_net import SimpleMetaNet
from src.model.utils.loss import linear_loss, linear_loss_multi
from src.utils.utils import SetType, \
    get_metric_name, Metric


def compute_metrics_for_all_sets(config: dict, meta_predictor: SimpleMetaNet, predictor: Predictor,
                                 criterion: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                                 test_loader: DataLoader, is_computing_test_bounds: bool) -> tuple[dict, dict, dict]:
    train_metrics = compute_metrics(config, meta_predictor, predictor, criterion, train_loader, False,
                                    set_type=SetType.TRAIN)
    valid_metrics = compute_metrics(config, meta_predictor, predictor, criterion, valid_loader,
                                    False, set_type=SetType.VALID)

    test_metrics = compute_metrics(config, meta_predictor, predictor, criterion, test_loader,
                                   is_computing_test_bounds, set_type=SetType.TEST)

    return train_metrics, valid_metrics, test_metrics


def compute_metrics(config: dict, meta_predictor: SimpleMetaNet, predictor: Predictor, criterion: nn.Module,
                    data_loader: DataLoader, are_bounds_computed: bool, set_type: SetType) -> dict[str, np.ndarray]:
    linear_bounds = []
    hyperparam_bounds = []
    kl_bounds = []
    marchand_bounds = []

    meta_predictor.eval()
    n_instances_per_class_per_dataset = meta_predictor.n_instances_per_class_per_dataset
    with torch.no_grad():
        n_instances_seen = 0
        n_batches = 0
        summed_losses_per_batch = []
        tot_acc = []
        for instances in data_loader:
            n_datasets = len(instances)
            n_instance_per_dataset = len(instances[0])
            n_instances_seen += n_datasets * n_instance_per_dataset // 2
            n_batches += 1

            instances = instances.float()
            targets = (instances[:, :, -config["target_size"]:] + 1) / 2
            targets = targets.float()

            if config["device"] == "gpu":
                instances = instances.cuda()
                targets = targets.cuda()
                meta_predictor = meta_predictor.cuda()

            meta_output = meta_predictor.forward(instances[:, :n_instances_per_class_per_dataset])
            predictor.set_params(meta_output)
            output = predictor(instances[:, n_instances_per_class_per_dataset:])
            loss = criterion(output[0], targets[:, n_instances_per_class_per_dataset:])
            summed_losses_per_batch.append(torch.sum(loss).cpu())

            if config["task"] == "classification":
                if config["target_size"] == 1:
                    acc = n_instances_per_class_per_dataset * linear_loss(output[1],
                                                                          targets[:,
                                                                          n_instances_per_class_per_dataset:] * 2 - 1)
                else:
                    acc = n_instances_per_class_per_dataset * linear_loss_multi(output[1],
                                                                                targets[:,
                                                                                n_instances_per_class_per_dataset:])
                #print(acc)
                #print(torch.mean(acc / n_instances_per_class_per_dataset).cpu())
                tot_acc.append(torch.mean(acc / n_instances_per_class_per_dataset).cpu())
                if are_bounds_computed:
                    for dataset_idx in range(len(instances)):
                        bounds = compute_bounds(["linear", "hyperparam", "kl", "marchand"], meta_predictor, predictor,
                                                n_instances_per_class_per_dataset,
                                                n_instances_per_class_per_dataset - acc.item(), 0.05, 0, 1,
                                                instances[[dataset_idx], n_instances_per_class_per_dataset:],
                                                targets[[dataset_idx], n_instances_per_class_per_dataset:],
                                                config["msg_size"], config["msg_type"], config["compression_set_size"])
                        linear_bounds.append(bounds[0])
                        hyperparam_bounds.append(bounds[1])
                        kl_bounds.append(bounds[2])
                        marchand_bounds.append(bounds[3])

        zero = np.array(0.0)
        if config["task"] == "classification":
            mean_accuracy = np.mean(tot_acc)
        elif config["task"] == "regression":
            mean_accuracy = zero
        else:
            raise NotImplementedError(f"The task '{config['task']}' is not supported.")

        mean_loss = np.sum(summed_losses_per_batch) / n_instances_seen

        if are_bounds_computed:
            mean_bounds = {Metric.LINEAR_BOUND.value: np.mean(linear_bounds),
                           Metric.HPARAM_BOUND.value: np.mean(hyperparam_bounds),
                           Metric.KL_BOUND.value: np.mean(kl_bounds),
                           Metric.MARCHAND_BOUND.value: np.mean(marchand_bounds)}
            return {get_metric_name(set_type, Metric.ACCURACY): mean_accuracy,
                    get_metric_name(set_type, Metric.LOSS): mean_loss, **mean_bounds}

        return {get_metric_name(set_type, Metric.ACCURACY): mean_accuracy,
                get_metric_name(set_type, Metric.LOSS): mean_loss}
