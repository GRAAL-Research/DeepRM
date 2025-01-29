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
                                    is_computing_test_bounds, set_type=SetType.VALID)

    test_metrics = compute_metrics(config, meta_predictor, predictor, criterion, test_loader,
                                   is_computing_test_bounds, set_type=SetType.TEST)

    return train_metrics, valid_metrics, test_metrics


def compute_metrics(config: dict, meta_predictor: SimpleMetaNet, predictor: Predictor, criterion: nn.Module,
                    data_loader: DataLoader, are_bounds_computed: bool, set_type: SetType) -> dict[str, np.ndarray]:
    linear_bounds = []
    hyperparam_bounds = []
    kl_bounds = []
    marchand_bounds = []
    kl_dis_bounds = []

    meta_predictor.eval()
    with torch.no_grad():
        n_instances_seen = 0
        n_batches = 0
        summed_query_losses_per_batch = []
        summed_support_losses_per_batch = []
        tot_acc = []
        for instances in data_loader:
            instances = instances[0]
            n_datasets = len(instances)
            n_instance_per_dataset = len(instances[0])
            n_instances_per_class_per_dataset = n_instance_per_dataset // 2
            n_instances_seen += n_datasets * n_instance_per_dataset // 2
            n_batches += 1

            instances = instances.float()
            targets = (instances[:, :, -config["target_size"]:] + 1) / 2
            targets = targets.float()

            if config["device"] == "gpu":
                instances = instances.cuda()
                targets = targets.cuda()
                meta_predictor = meta_predictor.cuda()

            meta_output = meta_predictor.forward(instances[:, :n_instances_per_class_per_dataset], is_in_test_mode=True)
            predictor.set_params(meta_output)
            support_output = predictor(instances[:, :n_instances_per_class_per_dataset])
            support_loss = criterion(support_output[0], targets[:, :n_instances_per_class_per_dataset])
            query_output = predictor(instances[:, n_instances_per_class_per_dataset:])
            query_loss = criterion(query_output[0], targets[:, n_instances_per_class_per_dataset:])
            summed_support_losses_per_batch.append(torch.sum(support_loss).cpu())
            summed_query_losses_per_batch.append(torch.sum(query_loss).cpu())

            if config["task"] == "classification":
                if config["target_size"] == 1:
                    support_acc = n_instances_per_class_per_dataset * linear_loss(support_output[1],
                                                                                targets[:,
                                                                                :n_instances_per_class_per_dataset] * 2 - 1)
                    query_acc = n_instances_per_class_per_dataset * linear_loss(query_output[1],
                                                                          targets[:,
                                                                          n_instances_per_class_per_dataset:] * 2 - 1)
                else:
                    support_acc = n_instances_per_class_per_dataset * torch.mean(linear_loss_multi(support_output[1],
                                                                                targets[:,
                                                                                :n_instances_per_class_per_dataset]), dim=-1)
                    query_acc = n_instances_per_class_per_dataset * torch.mean(linear_loss_multi(query_output[1],
                                                                                targets[:,
                                                                                n_instances_per_class_per_dataset:]), dim=-1)
                tot_acc.append(torch.mean(query_acc / n_instances_per_class_per_dataset).cpu())
                if are_bounds_computed:
                    for dataset_idx in range(len(instances)):
                        bounds = compute_bounds(["linear", "hyperparam", "kl", "marchand", "kl_dis"], meta_predictor, predictor,
                                                n_instances_per_class_per_dataset,
                                                n_instances_per_class_per_dataset - support_acc[dataset_idx].item(), 0.10, 0, 1,
                                                instances[[dataset_idx], :n_instances_per_class_per_dataset],
                                                targets[[dataset_idx], :n_instances_per_class_per_dataset],
                                                config["msg_size"], config["msg_type"], config["compression_set_size"],
                                                config["compression_pool_size"])
                        linear_bounds.append(bounds[0])
                        hyperparam_bounds.append(bounds[1])
                        kl_bounds.append(bounds[2])
                        marchand_bounds.append(bounds[3])
                        kl_dis_bounds.append(bounds[4])
        zero = np.array(0.0)
        if config["task"] == "classification":
            mean_accuracy = np.mean(tot_acc)
            std_accuracy = torch.std(query_acc / n_instances_per_class_per_dataset)
        elif config["task"] == "regression":
            mean_accuracy = zero
        else:
            raise NotImplementedError(f"The task '{config['task']}' is not supported.")
        mean_loss = np.sum(summed_query_losses_per_batch) / n_instances_seen
        if are_bounds_computed:
            if set_type is SetType.VALID:
                bounds = {Metric.VALID_LINEAR_BOUND_MEAN.value: np.mean(linear_bounds),
                          Metric.VALID_HPARAM_BOUND_MEAN.value: np.mean(hyperparam_bounds),
                          Metric.VALID_KL_BOUND_MEAN.value: np.mean(kl_bounds),
                          Metric.VALID_KL_DISINTEGRATED_BOUND_MEAN.value: np.mean(kl_dis_bounds),
                          Metric.VALID_MARCHAND_BOUND_MEAN.value: np.mean(marchand_bounds),

                          Metric.VALID_LINEAR_BOUND_STD.value: np.std(linear_bounds),
                          Metric.VALID_HPARAM_BOUND_STD.value: np.std(hyperparam_bounds),
                          Metric.VALID_KL_BOUND_STD.value: np.std(kl_bounds),
                          Metric.VALID_KL_DISINTEGRATED_BOUND_STD.value: np.std(kl_dis_bounds),
                          Metric.VALID_MARCHAND_BOUND_STD.value: np.std(marchand_bounds)}
            elif set_type is SetType.TEST:
                bounds = {Metric.TEST_LINEAR_BOUND_MEAN.value: np.mean(linear_bounds),
                          Metric.TEST_HPARAM_BOUND_MEAN.value: np.mean(hyperparam_bounds),
                          Metric.TEST_KL_BOUND_MEAN.value: np.mean(kl_bounds),
                          Metric.TEST_KL_DISINTEGRATED_BOUND_MEAN.value: np.mean(kl_dis_bounds),
                          Metric.TEST_MARCHAND_BOUND_MEAN.value: np.mean(marchand_bounds),

                          Metric.TEST_LINEAR_BOUND_STD.value: np.std(linear_bounds),
                          Metric.TEST_HPARAM_BOUND_STD.value: np.std(hyperparam_bounds),
                          Metric.TEST_KL_BOUND_STD.value: np.std(kl_bounds),
                          Metric.TEST_KL_DISINTEGRATED_BOUND_STD.value: np.std(kl_dis_bounds),
                          Metric.TEST_MARCHAND_BOUND_STD.value: np.std(marchand_bounds)}
            return {get_metric_name(set_type, Metric.ACCURACY_MEAN): mean_accuracy,
                    get_metric_name(set_type, Metric.ACCURACY_STD): std_accuracy,
                    get_metric_name(set_type, Metric.LOSS): mean_loss, **bounds}

        return {get_metric_name(set_type, Metric.ACCURACY_MEAN): mean_accuracy,
                get_metric_name(set_type, Metric.ACCURACY_STD): std_accuracy,
                get_metric_name(set_type, Metric.LOSS): mean_loss}
