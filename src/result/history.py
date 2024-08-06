import wandb


def add_new_values_to_history(history: dict[str, list], new_values: dict) -> None:
    for key, new_value in new_values.items():
        if key not in history:
            raise ValueError(f"The key '{key}' is not in history.")
        history[key].append(new_value)


def log_history_in_wandb(history: dict[str, list]) -> None:
    recent_value_idx = -1
    most_recent_history = {metric: values[recent_value_idx] for metric, values in history.items()}
    wandb.log(most_recent_history)
