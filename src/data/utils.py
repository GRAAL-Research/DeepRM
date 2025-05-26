from pathlib import Path

import numpy as np

DATA_BASE_PATH = Path("dataset")

def create_empty_datasets(config: dict) -> np.ndarray:
    return np.zeros(
        (config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] + config["target_size"]))


def shuffled_x_and_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    random_indices = np.random.permutation(len(x))

    return x[random_indices], y[random_indices]
