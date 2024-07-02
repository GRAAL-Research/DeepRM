import numpy as np

from dataset.datasets_utils import create_empty_datasets, shuffled_x_and_y


def generate_blob_datasets(config: dict) -> np.ndarray:
    datasets = create_empty_datasets(config)

    for dataset_idx in range(len(datasets)):
        if config["is_blob_identical"]:
            current_seed_value = np.random.get_state()[1][0]
            np.random.seed(config["seed"])

        datasets[dataset_idx] = generate_random_blob_dataset(config)

        if config["is_blob_identical"]:
            np.random.seed(current_seed_value)

    return datasets


def generate_random_blob_dataset(config: dict) -> np.ndarray:
    nb_of_samples_of_the_second_class = config["m"] // 2

    x_of_first_class = generate_random_blob_features(config)
    y_of_first_class = np.ones((config["m"], 1))
    x, y = modify_first_class_samples_to_add_the_second_class(x_of_first_class, y_of_first_class,
                                                              nb_of_samples_of_the_second_class)

    if config["shuffle_each_dataset_samples"]:
        x, y = shuffled_x_and_y(x, y)

    return np.hstack((x, y))


def generate_random_blob_features(config: dict) -> np.ndarray:
    random_gaussian_blob_center = np.random.rand(config["d"]) * 10 - 5
    return np.random.multivariate_normal(mean=random_gaussian_blob_center, cov=np.eye(config["d"]),
                                         size=config["m"])


def modify_first_class_samples_to_add_the_second_class(x: np.ndarray, y: np.ndarray,
                                                       nb_of_samples_of_the_second_class: int) -> tuple[
        np.ndarray, np.ndarray]:
    nb_of_features = x.shape[1]
    x[: nb_of_samples_of_the_second_class] += np.sign(np.random.rand(nb_of_features) - 0.5) * 5
    y[: nb_of_samples_of_the_second_class] -= 2

    return x, y
