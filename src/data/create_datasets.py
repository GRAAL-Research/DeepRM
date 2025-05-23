import numpy as np

from src.data.dataset.blob import generate_blob_datasets
from src.data.dataset.mnist_binary import load_mnist_binary
from src.data.dataset.mnist_multi import load_mnist_multi
from src.data.dataset.mnist_label import load_mnist_label
from src.data.dataset.cifar100 import load_cifar100
from src.data.dataset.mnist import load_mnist
from src.data.dataset.moon import generate_moon_datasets
from src.data.dataset.moon_and_blob import generate_moon_and_blob_datasets
from src.data.dataset.mtpl import load_MTPL


def create_datasets(config: dict) -> np.ndarray:
    if config["dataset"] == "moon":
        assert config["n_features"] == 2
        return generate_moon_datasets(config)

    elif config["dataset"] == "blob":
        assert config["n_features"] == 2
        return generate_blob_datasets(config)

    elif config["dataset"] in ["moon_and_blob", "blob_and_moon"]:
        assert config["n_features"] == 2
        return generate_moon_and_blob_datasets(config)

    elif config["dataset"] in ["mnist_17", "mnist_49", "mnist_56"]:
        return load_mnist_binary(config, config["dataset"][-2], config["dataset"][-1])

    elif config["dataset"] == "mnist_multi":
        return load_mnist_multi(config)

    elif config["dataset"] == "mnist_label":
        return load_mnist_label(config)

    elif config["dataset"] == "cifar100":
        return load_cifar100(config)

    elif config["dataset"] == "mnist":
        return load_mnist(config)

    elif config["dataset"] in ["MTPL2_frequency", "MTPL2_severity", "MTPL2_pure"]:
        return load_MTPL(config["dataset"][6:], config["n_dataset"], config["n_instances_per_dataset"])

    raise NotImplementedError(f"The dataset '{config['dataset']}' is not supported.")
