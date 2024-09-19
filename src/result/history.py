import wandb

from src.utils.utils import (TEST_ACCURACY_MEAN, TRAIN_ACCURACY_MEAN, VALID_ACCURACY_MEAN, TEST_ACCURACY_STD,
                             TRAIN_ACCURACY_STD, VALID_ACCURACY_STD, TEST_LOSS, TRAIN_LOSS, VALID_LOSS, Metric)


def create_history() -> dict[str, list]:
    return {TRAIN_LOSS: [], VALID_LOSS: [], TEST_LOSS: [], TRAIN_ACCURACY_MEAN: [],
            VALID_ACCURACY_MEAN: [], TEST_ACCURACY_MEAN: [], TRAIN_ACCURACY_STD: [],
            VALID_ACCURACY_STD: [], TEST_ACCURACY_STD: [], Metric.LINEAR_BOUND_MEAN.value: [],
            Metric.HPARAM_BOUND_MEAN.value: [], Metric.KL_BOUND_MEAN.value: [], Metric.MARCHAND_BOUND_MEAN.value: [],
            Metric.LINEAR_BOUND_STD.value: [],
            Metric.HPARAM_BOUND_STD.value: [], Metric.KL_BOUND_STD.value: [], Metric.MARCHAND_BOUND_STD.value: []
            }


def update_history(history: dict[str, list], new_values: dict) -> None:
    for key, new_value in new_values.items():
        if key not in history:
            raise ValueError(f"The key '{key}' is not in history.")
        history[key].append(new_value)


def log_history_in_wandb(history: dict[str, list]) -> None:
    recent_value_idx = -1
    most_recent_history = {metric: values[recent_value_idx] for metric, values in history.items()}
    wandb.log(most_recent_history)
