import numpy as np

from src.data.dataset.blob import generate_blob_datasets
from src.data.dataset.moon import generate_moon_datasets
from src.data.dataset.moon_and_blob import generate_moon_and_blob_datasets
from src.data.dataset.torchvision_dataset.cifar100 import CIFAR100
from src.data.dataset.torchvision_dataset.mnist import Mnist
from src.data.dataset.torchvision_dataset.mnist_binary import MnistBinary
from src.data.dataset.torchvision_dataset.mnist_binary_with_only_two_classes import MnistBinaryWithOnlyTwoClasses
from src.data.dataset.torchvision_dataset.mnist_label_shuffle import MnistLabelShuffle
from src.data.torchvision_dataset_creator import TorchvisionDatasetCreator


def create_datasets(config: dict) -> np.ndarray:
    if config["is_encoding_as_images"]:
        if config["dataset"] == "mnist":
            torchvision_dataset = Mnist()
        elif config["dataset"] == "mnist_binary_with_only_two_classes":
            torchvision_dataset = MnistBinaryWithOnlyTwoClasses()
        elif config["dataset"] == "mnist_label_shuffle":
            torchvision_dataset = MnistLabelShuffle()
        elif config["dataset"] == "mnist_binary":
            torchvision_dataset = MnistBinary()
        elif config["dataset"] == "cifar100":
            torchvision_dataset = CIFAR100()
        else:
            raise NotImplementedError(f"The image dataset '{config['dataset']}' is not supported.")

        torchvision_dataset_creator = TorchvisionDatasetCreator(torchvision_dataset)
        return torchvision_dataset_creator.create_meta_dataset(config)

    if config["dataset"] == "moon":
        assert config["n_features"] == 2
        return generate_moon_datasets(config)

    elif config["dataset"] == "blob":
        assert config["n_features"] == 2
        return generate_blob_datasets(config)

    elif config["dataset"] in ["moon_and_blob", "blob_and_moon"]:
        assert config["n_features"] == 2
        return generate_moon_and_blob_datasets(config)

    raise NotImplementedError(f"The dataset '{config['dataset']}' is not supported.")
