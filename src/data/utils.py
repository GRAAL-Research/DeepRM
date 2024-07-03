import numpy as np


def create_empty_datasets(config: dict) -> np.ndarray:
    y_target_second_dim = 1
    return np.zeros((config["n_dataset"], config["m"], config["n_features"] + y_target_second_dim))


def shuffled_x_and_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    random_indices = np.random.permutation(len(x))

    return x[random_indices], y[random_indices]
