from copy import copy
from time import time

import torch
import wandb
from loguru import logger

from src.data.create_datasets import create_datasets
from src.data.loaders import train_valid_loaders
from src.model.predictor.create_predictor import create_predictor
from src.result.compute_stats import compute_accuracy_loss_and_bounds
from src.result.decision_boundaries import show_decision_boundaries
from src.result.history import update_hist, update_wandb
from src.result.performance_matrix import show_performance_matrix
from src.training.criterion import create_criterion
from src.training.message_penalty import create_message_penalty_function
from src.training.meta_predictor import create_meta_predictor
from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler
from src.utils.epoch_logger import EpochLogger
from src.utils.utils import create_run_name


def train_meta_predictor(config: dict, is_sending_wandb_last_run_alert: bool) -> tuple[dict, int]:
    """
    Returns:
        tuple of: information about the model at the best training epoch (dictionary), best training epoch (int).
    """
    torch.autograd.set_detect_anomaly(True)

    datasets = create_datasets(config)
    predictor = create_predictor(config)
    meta_predictor = create_meta_predictor(config, predictor)
    criterion = create_criterion(config)
    message_penalty_function = create_message_penalty_function(config)
    optimizer = create_optimizer(config, meta_predictor)
    scheduler = create_scheduler(config, optimizer)

    valid_metric = "valid_acc" if config["task"] == "classification" else "valid_loss"
    n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
    train_loader, valid_loader, test_loader, tr_var, vd_var, te_var, idx = train_valid_loaders(datasets,
                                                                                               config["batch_size"],
                                                                                               config["splits"],
                                                                                               seed=config["seed"])
    best_rolling_val_acc = 0
    best_epoch = 0
    # The following information will be recorded at each epoch
    hist = {"epoch": [], "train_loss": [], "valid_loss": [], "test_loss": [], "train_acc": [], "valid_acc": [],
            "test_acc": [], "bound_lin": [], "bound_hyp": [], "bound_kl": [], "bound_mrch": []}

    if config["is_using_wandb"]:
        run_name = create_run_name(config)
        wandb.init(name=run_name, project=config["project_name"], config=config)

    start_time = time()
    for epoch_idx in range(config["max_epoch"]):
        meta_predictor.train()
        with (torch.enable_grad()):
            for instances in train_loader:
                instances = instances.float()[:, n_instances_per_class_per_dataset:]
                targets = (instances[:, :, -1] + 1) / 2
                instances = instances.float()
                targets = targets.float()

                if config["device"] == "gpu":
                    instances = instances.cuda()
                    targets = targets.cuda()
                    meta_predictor = meta_predictor.cuda()

                optimizer.zero_grad()
                meta_output = meta_predictor.forward(instances)
                predictor.set_params(meta_output)
                output, _ = predictor.forward(instances)

                if config["is_dataset_balanced"] or config["task"] == "regression":
                    loss = torch.mean(torch.mean(criterion(output, targets), dim=1) ** config["loss_exponent"])
                else:
                    loss = 0
                    for batch in range(len(output)):
                        loss += (torch.mean(criterion(output[batch, targets[batch] == 0],
                                                      targets[batch, targets[batch] == 0])) / 2 +
                                 torch.mean(criterion(output[batch, targets[batch] == 1],
                                                      targets[batch, targets[batch] == 1])) / 2) ** config[
                                    "loss_exponent"]
                    loss /= len(output)
                if config["msg_type"] is not None and config["msg_size"] > 0:
                    loss += message_penalty_function(meta_predictor.get_message(), config["msg_penalty_coef"])
                loss.backward()
                optimizer.step()
        # Computation of statistics about the current training epoch
        train_accuracy, train_loss, _ = compute_accuracy_loss_and_bounds(config, meta_predictor, predictor, criterion,
                                                                         train_loader)
        valid_accuracy, valid_loss, _ = compute_accuracy_loss_and_bounds(config, meta_predictor, predictor, criterion,
                                                                         valid_loader)
        test_accuracy, test_loss, bound = compute_accuracy_loss_and_bounds(config, meta_predictor, predictor, criterion,
                                                                           test_loader)
        hist_values = (
            train_accuracy, train_loss, valid_accuracy, valid_loss, test_accuracy, test_loss, bound, config["msg_size"],
            config["compression_set_size"], epoch_idx
        )
        update_hist(hist, hist_values)
        rolling_val_perf = torch.mean(torch.tensor(hist[valid_metric][-min(100, epoch_idx + 1):]))

        if config["is_using_wandb"]:
            update_wandb(wandb, hist)

        epo = "0" * (epoch_idx + 1 < 100) + "0" * (epoch_idx + 1 < 10) + str(epoch_idx + 1)

        if config["is_bound_computed"]:
            bound_info_to_log = (f" - bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), "
                                 f"(marchand: {bound[3]:.2f})")
        else:
            bound_info_to_log = ""

        time_info_to_log = f" - time: {round(time() - start_time)}s"
        if config["task"] == "classification":
            EpochLogger.log(
                f"epoch {epo} - train_acc: {train_accuracy:.3f} - val_acc: {valid_accuracy:.3f}"
                f" - test_acc: {test_accuracy:.3f}"
                f"{bound_info_to_log}{time_info_to_log}")
        elif config["task"] == "regression":
            EpochLogger.log(
                f"Epoch {epo} - Train R2: {1 - train_loss / tr_var:.4f} - Val R2: {1 - valid_loss / vd_var:.4f}"
                f" - Test R2: {1 - test_loss / te_var:.4f}"
                f"{bound_info_to_log}{time_info_to_log}")
        else:
            raise NotImplementedError(f"The task '{config['task']}' is not supported.")

        scheduler.step(rolling_val_perf)

        has_significant_improvement_being_done_in_validation = rolling_val_perf > best_rolling_val_acc + config[
            "early_stopping_tolerance"]
        if has_significant_improvement_being_done_in_validation:
            best_epoch = epoch_idx
            best_rolling_val_acc = copy(rolling_val_perf)

        has_no_learning_being_made = train_accuracy < 0.525 and epoch_idx > 50
        has_no_improvement_for_a_while = epoch_idx - best_epoch > config["early_stopping_patience"]
        if has_no_learning_being_made or has_no_improvement_for_a_while:
            logger.info("The early stopping stopped the training.")
            break

    if config["is_media_computed"]:
        if config["n_features"] == 2 and config["is_using_wandb"]:
            show_decision_boundaries(meta_predictor, config["dataset"], test_loader, predictor, wandb, config["device"])
        if config["dataset"] in ["mnist", "cifar100_binary"]:
            show_performance_matrix(meta_predictor, predictor, config["dataset"], datasets, idx, config["n_dataset"],
                                    config["is_using_wandb"], wandb, config["batch_size"], config["device"])

    if config["is_using_wandb"]:
        if is_sending_wandb_last_run_alert and config["is_wandb_alert_activated"]:
            wandb.alert(title="✅ Done", text="The experiment is over.")
        wandb.finish()

    return hist, best_epoch
