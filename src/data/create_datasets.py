import numpy as np
import math

from src.data.dataset.blob import generate_blob_datasets
from src.data.dataset.mnist import load_mnist, load_mnist_labels
from src.data.dataset.cifar100 import load_cifar100, load_cifar100_labels
from src.data.dataset.moon import generate_moon_datasets
from src.data.dataset.moon_and_blob import generate_moon_and_blob_datasets
from src.data.dataset.mtpl import load_MTPL


def create_datasets(config: dict) -> tuple[np.ndarray, list]:
    if config["dataset"] == "moon":
        assert config["n_features"] == 2
        return generate_moon_datasets(config), ["-1", "+1"]

    elif config["dataset"] == "blob":
        assert config["n_features"] == 2
        return generate_blob_datasets(config), ["-1", "+1"]

    elif config["dataset"] in ["moon_and_blob", "blob_and_moon"]:
        assert config["n_features"] == 2
        return generate_moon_and_blob_datasets(config), ["-1", "+1"]

    elif config["dataset"] == "mnist":
        return load_mnist(config), load_mnist_labels()[:int((1 + math.sqrt(1 + 4 * int(config["n_dataset"]))) / 2)]

    elif config["dataset"] == "cifar100_binary":
        return load_cifar100(config), load_cifar100_labels()[:int((1 + math.sqrt(1 + 4 * int(config["n_dataset"]))) /2)]

    elif config["dataset"] in ["MTPL2_frequency", "MTPL2_severity", "MTPL2_pure"]:
        return load_MTPL(config["dataset"][6:], config["n_dataset"], config["n_instances_per_dataset"]), []

    raise NotImplementedError(f"The dataset '{config['dataset']}' is not supported.")
