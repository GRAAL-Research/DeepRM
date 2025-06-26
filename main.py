import wandb
from loguru import logger

from src.config.config import create_hyperparameters_config
from src.config.grid_search_config import create_config_combinations_sorted_by_dataset
from src.training.train import train_meta_predictor
from src.utils.default_logger import DefaultLogger
from src.utils.run_completion import is_run_already_completed, label_run_config_as_completed
from src.utils.utils import set_random_seed, create_run_name


def main(config_combinations: list[dict]) -> None:
    n_runs = len(config_combinations)

    # We loop over all of the tested configurations combinations
    for run_idx, config in enumerate(config_combinations):
        logger.info(f"Launching run {run_idx + 1}/{n_runs} : {config}")

        # Tests to verify whether it is relevant to run the run
        if config["msg_size"] == 0 and config["compression_set_size"] == 0 and config["msg_type"] == 'cnt':
            logger.info("Skipping the run... Opaque network.")
            continue
        if config["msg_type"] == "dsc" and config["msg_penalty_coef"] > 0:
            logger.info("Skipping the run... It doesn't make sens to regularize discrete messages.")
            continue
        if is_run_already_completed(config):
            logger.info("Skipping the run... It is already completed.")
            continue
        if config['deepset_dim'][-1] % config['target_size'] != 0:
            logger.info("Skipping the run... KME output size and target size are incompatible.")
            continue

        if config["is_using_wandb"]:
            run_name = create_run_name(config)
            wandb.init(name=run_name, project=config["project_name"], config=config)

        set_random_seed(config["seed"])
        # Main function: launches the training loop
        train_meta_predictor(config)

        is_the_last_run = run_idx + 1 == n_runs
        if config["is_using_wandb"]:
            wandb.finish()
            if is_the_last_run and config["is_wandb_alert_activated"]:
                wandb.alert(title="✅ Done", text="The experiment is over.")

        # Run results are saved locally
        if config["is_saving_completed_runs_locally"]:
            label_run_config_as_completed(config)


if __name__ == "__main__":
    DefaultLogger.apply_format()
    loaded_config = create_hyperparameters_config()
    config_combinations = create_config_combinations_sorted_by_dataset(loaded_config)
    main(config_combinations)
