from copy import copy
from time import time

import torch
import wandb
from loguru import logger

from src.data.create_datasets import create_datasets
from src.data.loaders import train_valid_loaders
from src.model.predictor import Predictor
from src.result.compute_stats import stats
from src.result.decision_boundaries import show_decision_boundaries
from src.result.history import update_hist, update_wandb
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
    pred = Predictor(config)
    meta_predictor = create_meta_predictor(config, pred)
    criterion = create_criterion(config)
    message_penalty_function = create_message_penalty_function(config)
    optimizer = create_optimizer(config, meta_predictor)
    scheduler = create_scheduler(config, optimizer)

    valid_metric = "valid_acc" if config["task"] == "classification" else "valid_loss"
    n_instances_per_class_per_dataset = config["n_instances_per_dataset"] // 2
    train_loader, valid_loader, test_loader, tr_var, vd_var, te_var = train_valid_loaders(datasets,
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
            for inputs in train_loader:  # Iterating over the various batches
                inputs = inputs.float()[:, n_instances_per_class_per_dataset:]
                targets = (inputs.clone()[:, :, -1] + 1) / 2
                inputs, targets = inputs.float(), targets.float()
                if config["device"] == "gpu":
                    inputs, targets, meta_predictor = inputs.cuda(), targets.cuda(), meta_predictor.cuda()
                optimizer.zero_grad()
                meta_output = meta_predictor(inputs)
                pred.set_weights(meta_output, len(inputs))
                output, _ = pred.forward(inputs)
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
                loss += message_penalty_function(meta_predictor.msg, config["msg_penalty_coef"])  # Regularized loss
                loss.backward()
                optimizer.step()
        # Computation of statistics about the current training epoch
        tr_acc, tr_loss, _ = stats(config["task"], meta_predictor, pred, criterion, train_loader, config["msg_type"],
                                   config["is_bound_computed"], config["device"])
        vd_acc, vd_loss, _ = stats(config["task"], meta_predictor, pred, criterion, valid_loader, config["msg_type"],
                                   config["is_bound_computed"], config["device"])
        te_acc, te_loss, bound = stats(config["task"], meta_predictor, pred, criterion, test_loader,
                                       config["msg_type"],
                                       config["is_bound_computed"],
                                       config["device"])
        update_hist(hist, (tr_acc, tr_loss, vd_acc, vd_loss, te_acc, te_loss, bound, config["msg_size"],
                           meta_predictor.compression_set_size, epoch_idx))  # Tracking results
        rolling_val_perf = torch.mean(torch.tensor(hist[valid_metric][-min(100, epoch_idx + 1):]))

        if config["is_using_wandb"]:
            update_wandb(wandb, hist)

        epo = "0" * (epoch_idx + 1 < 100) + "0" * (epoch_idx + 1 < 10) + str(epoch_idx + 1)

        if config["task"] == "classification":
            EpochLogger.log(
                f"Epoch {epo} - Train acc: {tr_acc:.4f} - Val acc: {vd_acc:.4f} - Test acc: {te_acc:.4f} - "
                f"Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), "
                f"(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - start_time)}")
        elif config["task"] == "regression":
            EpochLogger.log(
                f"Epoch {epo} - Train R2: {1 - tr_loss / tr_var:.4f} - Val R2: {1 - vd_loss / vd_var:.4f} - "
                f"Test R2: {1 - te_loss / te_var:.4f} - "
                f"Bounds: (lin: {bound[0]:.2f}), (hyp: {bound[1]:.2f}), (kl: {bound[2]:.2f}), "
                f"(Marchand: {bound[3]:.2f}) - Time (s): {round(time() - start_time)}")
        else:
            raise NotImplementedError(f"The task '{config['task']}' is not supported.")

        scheduler.step(rolling_val_perf)

        has_significant_improvement_being_done_in_validation = rolling_val_perf > best_rolling_val_acc + config[
            "early_stopping_tolerance"]
        if has_significant_improvement_being_done_in_validation:
            best_epoch = epoch_idx
            best_rolling_val_acc = copy(rolling_val_perf)

        has_no_learning_being_made = tr_acc < 0.525 and epoch_idx > 50
        has_no_improvement_for_a_while = epoch_idx - best_epoch > config["early_stopping_patience"]
        if has_no_learning_being_made or has_no_improvement_for_a_while:
            logger.info("The early stopping stopped the training.")
            break

    if config["n_features"] == 2 and config["is_using_wandb"]:
        show_decision_boundaries(meta_predictor, config["dataset"], test_loader, pred, wandb, config["device"])

    if config["is_using_wandb"]:
        if is_sending_wandb_last_run_alert and config["is_wandb_alert_activated"]:
            wandb.alert(title="✅ Done", text="The experiment is over.")
        wandb.finish()

    return hist, best_epoch
