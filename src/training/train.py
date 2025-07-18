import time
from copy import copy

import torch
from loguru import logger

from src.data.create_meta_dataset import create_datasets
from src.data.loaders import train_valid_and_test_indices, create_data_loader, compute_variances
from src.model.mlp import MLP
from src.model.predictor.create_predictor import create_predictor
from src.result.history import update_history, log_history_in_wandb, create_history
from src.training.compute_medias import compute_medias
from src.training.compute_metrics import compute_metrics_for_all_sets
from src.training.early_stopping import is_stopping_training
from src.training.factory.criterion import create_criterion
from src.training.factory.meta_predictor import create_meta_predictor
from src.training.factory.optimizer import create_optimizer
from src.training.factory.scheduler import create_scheduler
from src.training.launch_epoch_training import launch_one_meta_training_epoch
from src.training.launch_prior_training import launch_prior_training
from src.training.log_epoch_in_terminal import log_epoch_info_in_terminal
from src.utils.utils import VALID_ACCURACY_MEAN, VALID_LOSS


def train_meta_predictor(config: dict) -> None:
    torch.autograd.set_detect_anomaly(True)

    datasets = create_datasets(config)
    predictor = create_predictor(config)
    meta_predictor = create_meta_predictor(config, predictor)
    criterion = create_criterion(config)
    optimizer = create_optimizer(config, meta_predictor)
    scheduler = create_scheduler(config, optimizer)

    validation_metric = VALID_ACCURACY_MEAN if config["task"] == "classification" else VALID_LOSS

    train_idx, valid_idx, test_idx = train_valid_and_test_indices(config["dataset"], datasets, config["splits"],
                                                                  config["are_test_classes_shared_with_train"],
                                                                  config["seed"])
    train_loader = create_data_loader(datasets, config["meta_batch_size"], train_idx)
    valid_loader = create_data_loader(datasets, config["meta_batch_size"], valid_idx)
    test_loader = create_data_loader(datasets, config["meta_batch_size"], test_idx)
    train_var, valid_var, test_var = compute_variances(datasets, train_idx, valid_idx, test_idx)

    best_rolling_val_acc = 0
    best_epoch = 0
    history = create_history()
    start_time = time.time()

    # If the config permits it, a "prior" is computed for the downstream predictor; the future predictions of the meta
    #   model will just fine-tune this "prior".
    if config["compute_prior"]:
        prior = MLP(config["n_features"], config["pred_hidden_sizes"] + [config["target_size"]], config["device"],
                    config["has_skip_connection"], config["has_batch_norm"],
                    config["init_scheme"], None)
        prior = launch_prior_training(config, prior, train_loader, test_loader, criterion)
        predictor.get_batch_norm_from_prior(prior)
        predictor.set_prior_weights(prior)

    for epoch_idx in range(config["max_epoch"]):
        predictor = launch_one_meta_training_epoch(config, meta_predictor, predictor, train_loader, criterion, optimizer)

        is_computing_test_bounds = False
        if config["is_bound_computed"] and epoch_idx % config["bound_computation_epoch_frequency"] == 0:
            is_computing_test_bounds = True

        train_metrics, valid_metrics, test_metrics = compute_metrics_for_all_sets(config, meta_predictor, predictor,
                                                                                  criterion,
                                                                                  train_loader, valid_loader,
                                                                                  test_loader,
                                                                                  is_computing_test_bounds)

        log_epoch_info_in_terminal(config, train_metrics, valid_metrics, test_metrics, train_var, valid_var,
                                   test_var, start_time, epoch_idx, is_computing_test_bounds)

        new_history_values = train_metrics | valid_metrics | test_metrics
        update_history(history, new_history_values)
        if config["is_using_wandb"]:
            log_history_in_wandb(history)

        # The rolling validation performance tells whether to stop the training.
        rolling_val_perf = compute_rolling_performance(history, validation_metric, epoch_idx)
        has_done_significant_improvement = rolling_val_perf > best_rolling_val_acc + config["early_stopping_tolerance"]
        if has_done_significant_improvement:
            best_epoch = epoch_idx
            best_rolling_val_acc = copy(rolling_val_perf)

        if is_stopping_training(config, train_metrics, epoch_idx, best_epoch, config["target_size"]):
            logger.info("The early stopping stopped the training.")
            break

        scheduler.step(rolling_val_perf)

    if config["is_media_computed"]:
        compute_medias(config, meta_predictor, test_loader, predictor)


def compute_rolling_performance(history: dict[str, list], metric: str, epoch_idx: int) -> torch.Tensor:
    max_epoch_on_which_the_rolling_performance_is_computed = 100
    return torch.mean(torch.tensor(history[metric][-min(rolling_performance_max_epoch, epoch_idx + 1):]))
