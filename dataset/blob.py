import numpy as np

from dataset.datasets_utils import create_empty_datasets, shuffled_x_and_y


def generate_blob_datasets(config: dict) -> np.ndarray:
    datasets = create_empty_datasets(config)

    for dataset_idx in range(len(datasets)):
        datasets[dataset_idx] = generate_random_blob_dataset(config)

    return datasets


def generate_random_blob_dataset(config: dict) -> np.ndarray:
    n_samples = config["m"] if config["m"] % 2 == 0 else config["m"] - 1  # TODO: Do we need balanced classes?
    nb_of_classes = 2
    nb_of_instance_per_class = n_samples // nb_of_classes

    x = generate_random_blob_features(config, nb_of_instance_per_class)
    y = np.ones((n_samples, 1))
    y[:nb_of_instance_per_class] -= 2

    if config["shuffle_each_dataset_samples"]:
        x, y = shuffled_x_and_y(x, y)

    return np.hstack((x, y))


def generate_random_blob_features(config: dict, nb_of_instance_per_class: int) -> np.ndarray:
    random_gaussian_blob_center = np.random.rand(config["d"]) * 10 - 5
    number_of_classes = 2

    x = np.random.multivariate_normal(mean=random_gaussian_blob_center, cov=np.eye(config["d"]),
                                      size=number_of_classes * nb_of_instance_per_class)
    x[:nb_of_instance_per_class] = apply_second_class_modifications(x[:nb_of_instance_per_class])

    return x


def apply_second_class_modifications(instances_of_second_class: np.ndarray) -> dict:
    nb_of_features = instances_of_second_class.shape[1]
    return instances_of_second_class + np.sign(np.random.rand(nb_of_features) - 0.5) * 5


