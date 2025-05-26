import math
import numpy as np
from sklearn.datasets import make_moons

from src.data.utils import create_empty_datasets, shuffled_x_and_y


def generate_moon_datasets(config: dict) -> np.ndarray:
    datasets = create_empty_datasets(config)

    for dataset_idx in range(len(datasets)):
        if config["is_moon_identical"]:
            current_seed_value = np.random.get_state()[1][0]
            np.random.seed(config["seed"])

        datasets[dataset_idx] = generate_moon_dataset(config)

        if config["is_moon_identical"]:
            np.random.seed(current_seed_value)

    return datasets


def generate_moon_dataset(config: dict) -> np.ndarray:
    x, y = make_moons(config["n_instances_per_dataset"], noise=0.08, random_state=config["seed"])
    y[y == 0] = -1

    if config["shuffle_each_dataset_samples"]:
        x, y = shuffled_x_and_y(x, y)

    x = apply_random_transformations(x)

    return np.hstack((x, y.reshape(-1, 1)))


def apply_random_transformations(x: np.ndarray) -> np.ndarray:
    random_scale = np.random.randint(3, 8)
    x *= random_scale

    random_origin = np.random.randint(-10, 10, 2)
    x += random_origin

    random_radian_angle = math.radians(np.random.randint(0, 360))
    return rotate_all_x(x, random_radian_angle)


def rotate_all_x(x: np.ndarray, radian_angle: float) -> np.ndarray:
    for i in range(len(x)):
        x[i, 0], x[i, 1] = rotate_counterclockwise(x[i], radian_angle)

    return x


def rotate_counterclockwise(point_to_rotate: tuple[float, float], radian_angle: float,
                            point_around_which_to_rotate: tuple[float, float] = (0, 0)) -> tuple[float, float]:
    x_diff = point_to_rotate[0] - point_around_which_to_rotate[0]
    y_diff = point_to_rotate[1] - point_around_which_to_rotate[1]
    cos_angle = math.cos(radian_angle)
    sin_angle = math.sin(radian_angle)

    x_rotation = cos_angle * x_diff - sin_angle * y_diff
    y_rotation = sin_angle * x_diff + cos_angle * y_diff

    rotated_x = point_around_which_to_rotate[0] + x_rotation
    rotated_y = point_around_which_to_rotate[1] + y_rotation

    return rotated_x, rotated_y
