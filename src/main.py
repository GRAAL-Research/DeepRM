from src.config.config import create_config
from src.config.grid_search_config import create_config_combinations_sorted_by_dataset
from src.data.create_datasets import create_datasets
from src.model.utils.loss import l1, l2
from src.result.prevent_running_completed_job import is_run_already_done, save_run_in_a_text_file
from src.train import *
from src.utils import set_random_seed
import torch.nn as nn


def main(config_combinations: list[dict]) -> None:
    """
    Args:
        dataset (list of str): datasets (choices: "mnist", "moon", "blob", "moon_and_blob",
                                                  "MTPL2_frequency", "MTPL2_severity", "MTPL2_pure");

        balanced (list of bool): whether to consider balanced datasets during forward pass losses computation;                                          
        n_features (list of int): dimension of each example;
        splits (list of [float, float, float]): train, valid and test proportion of the data;
        meta_pred (list of str): meta_predictor to use (choices: "simplenet");
        pred_arch (list of list of int): architecture of the predictor (a ReLU network);
        comp_set_size (list of int): compression set size;
        ca_dim (list of list of int): custom attention"s MLP architecture;
        mod_1_dim (list of list of int): MLP #1 architecture;
        mod_2_dim (list of list of int): MLP #2 architecture;
        tau (list of int): temperature parameter (softmax in custom attention);
        criterion (list of str): loss function (choices: "bce_loss");
        loss_power (list of floats): power to be applied, by dataset;
        pen_msg (list of str): type of message penalty (choices: "l1", "l2");
        pen_msg_coef (list of float): message penalty coefficient;
        batch_size (list of int): batch size;
        factor (list of float): factor by which the learning rate is multiplied (one decay step);
        optimizer (list of str): meta-neural network optimizer (choices: "adam", "rmsprop");
        n_epoch (list of int): maximum number of epoch for the training phase;z
    """
    n_tasks = len(config_combinations)
    meta_pred, datasets, opti, scheduler, crit, penalty_msg = None, None, None, None, None, None
    is_sending_wandb_last_run_alert = False

    for i, config in enumerate(config_combinations):
        if config["dataset"] == "mnist":
            # For non-synthetic data, these are fixed
            config["criterion"] = "bce_loss"
            config["n_dataset"] = 90
            config["n_instances_per_dataset"] = 6313 * 2
            config["n_features"] = 784
            config["balanced"] = True
            config["task"] = "classification"
        elif config["dataset"] in ["MTPL2_frequency", "MTPL2_severity", "MTPL2_pure"]:
            # For non-synthetic data, these are fixed
            config["criterion"] = "mse_loss"
            config["n_features"] = 76
            config["batch_size"] = 50
            config["balanced"] = False
            config["task"] = "regression"
        elif config["dataset"] == "moon":
            config["criterion"] = "bce_loss"
            config["balanced"] = True
            config["task"] = "classification"

        if config["msg_size"] == 0:
            config["msg_type"] = "none"

        print(f"Launching task {i + 1}/{n_tasks} : {config}\n")

        if config["msg_type"] == "dsc" and config["pen_msg_coef"] > 0:  # Passes on incoherent hyp. param. comb.
            print("Doesn't make sens to regularize discrete messages; passing...\n")
        elif is_run_already_done(config):
            print("Already done; passing...\n")
        else:  # The current hyp. param. comb. will be tested
            set_random_seed(config["seed"])  # Sets the random seed for numpy, torch and random packages

            datasets = create_datasets(config)

            pred = Predictor(config)
            if config["meta_pred"] == "simplenet":  # Meta-predictor initialization
                meta_pred = SimpleMetaNet(pred.n_param, config)

            if config["criterion"] == "bce_loss":  # Criterion initialization
                crit = nn.BCELoss(reduction="none")
            if config["criterion"] == "mse_loss":
                crit = nn.MSELoss(reduction="none")

            if config["pen_msg"] == "l1":  # Message penalty initialization
                penalty_msg = l1
            elif config["pen_msg"] == "l2":
                penalty_msg = l2

            if config["optimizer"] == "adam":  # Optimizer initialization
                opti = torch.optim.Adam(meta_pred.parameters(), lr=copy(config["lr"]))
            elif config["optimizer"] == "rmsprop":
                opti = torch.optim.RMSprop(meta_pred.parameters(), lr=copy(config["lr"]))

            if config["scheduler"] == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opti, mode="max",
                                                                       factor=config["factor"],
                                                                       patience=config["scheduler_patience"],
                                                                       threshold=config["scheduler_threshold"],
                                                                       verbose=True)

            if i + 1 == n_tasks:
                is_sending_wandb_last_run_alert = True

            hist, best_epoch = train(meta_pred, pred, datasets, opti, scheduler, crit, penalty_msg, config,
                                     is_sending_wandb_last_run_alert)
            if config["is_saving_completed_runs_locally"]:
                save_run_in_a_text_file(config, hist, best_epoch)


if __name__ == "__main__":
    config_name = "config.yaml"

    loaded_config = create_config(config_name)

    config_combinations = create_config_combinations_sorted_by_dataset(loaded_config)
    main(config_combinations)
