import numpy as np


def create_empty_datasets(config: dict) -> np.ndarray:
    y_target_first_dim = 1
    return np.zeros((config["n_dataset"], config["m"], config["d"] + y_target_first_dim))


def shuffled_x_and_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx_to_randomize = np.arange(len(x))  # TODO is it right?
    np.random.shuffle(idx_to_randomize)

    return x[idx_to_randomize], y[idx_to_randomize]
