import wandb
from loguru import logger

from src.config.config import create_config
from src.config.grid_search_config import create_config_combinations_sorted_by_dataset
from src.training.train import train_meta_predictor
from src.utils.default_logger import DefaultLogger
from src.utils.run_completion import is_run_already_completed, label_run_config_as_completed
from src.utils.utils import set_random_seed, create_run_name


def main(config_combinations: list[dict]) -> None:
    n_runs = len(config_combinations)
    is_sending_wandb_last_run_alert = False

    for run_idx, config in enumerate(config_combinations):
        logger.info(f"Launching run {run_idx + 1}/{n_runs} : {config}")

        if config["msg_type"] == "dsc" and config["msg_penalty_coef"] > 0:
            logger.info("Skipping the run... It doesn't make sens to regularize discrete messages.")
            continue
        if is_run_already_completed(config):
            logger.info("Skipping the run... It is already done.")
            continue

        if config["is_using_wandb"]:
            run_name = create_run_name(config)
            wandb.init(name=run_name, project=config["project_name"], config=config)

        set_random_seed(config["seed"])

        is_the_last_run = run_idx + 1 == n_runs
        if is_the_last_run:
            is_sending_wandb_last_run_alert = True

        train_meta_predictor(config, is_sending_wandb_last_run_alert)

        if config["is_saving_completed_runs_locally"]:
            label_run_config_as_completed(config)


if __name__ == "__main__":
    try:
        DefaultLogger.apply_format()
        loaded_config = create_config()
        config_combinations = create_config_combinations_sorted_by_dataset(loaded_config)
        main(config_combinations)
    except Exception as error:
        wandb.alert(title="❌ Error", text="The experiment failed.")
        raise error
