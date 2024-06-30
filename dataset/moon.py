import math

import numpy as np
from sklearn.datasets import make_moons

from dataset.datasets_utils import create_empty_datasets

Position = tuple[float, float]


def generate_moon_datasets(config: dict) -> np.ndarray:
    datasets = create_empty_datasets(config)

    for dataset_idx in range(len(datasets)):
        datasets[dataset_idx] = generate_moon_dataset(config)

    return datasets


def generate_moon_dataset(config: dict) -> np.ndarray:
    n_samples = config["m"] if config["m"] % 2 == 0 else config["m"] - 1  # TODO: Do we need balanced classes?

    x, y = make_moons(n_samples, shuffle=config["shuffle_each_dataset_samples"], noise=0.08,
                      random_state=config["seed"])  # TODO WARNING the random state fix completely the dataset
    y[y == 0] = -1

    if not config["is_keeping_moon_identical"]:
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
    for i in range(len(x)):  # TODO : Is it right?
        x[i, 0], x[i, 1] = rotate_counter_clockwise(x[i], radian_angle)

    return x


def rotate_counter_clockwise(point_to_rotate: Position, radian_angle: float,
                             point_around_which_to_rotate: Position = (0, 0)) -> Position:
    x_diff = point_to_rotate[0] - point_around_which_to_rotate[0]
    y_diff = point_to_rotate[1] - point_around_which_to_rotate[1]
    cos_angle = math.cos(radian_angle)
    sin_angle = math.sin(radian_angle)

    x_rotation = cos_angle * x_diff - sin_angle * y_diff
    y_rotation = sin_angle * x_diff + cos_angle * y_diff

    rotated_x = point_around_which_to_rotate[0] + x_rotation
    rotated_y = point_around_which_to_rotate[1] + y_rotation

    return rotated_x, rotated_y
