import numpy as np


def create_empty_datasets(config: dict) -> np.ndarray:
    return np.zeros(
        (config["n_dataset"], config["n_instances_per_dataset"], config["n_features"] + config["target_size"]))


def shuffled_x_and_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    random_indices = np.random.permutation(len(x))

    return x[random_indices], y[random_indices]

def validate_n_features_for_images(config: dict) -> None:
    assert config["n_features"] == config["input_shape"][-1] * config["input_shape"][-2]
