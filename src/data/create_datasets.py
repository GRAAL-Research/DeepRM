from src.data.dataset.blob import generate_blob_datasets
from src.data.dataset.mnist import load_mnist
from src.data.dataset.moon import generate_moon_datasets
from src.data.dataset.moon_and_blob import generate_moon_and_blob_datasets
from src.data.dataset.mtpl import load_MTPL


def create_datasets(config: dict):
    if config["dataset"] == "moon":
        assert config["d"] == 2
        return generate_moon_datasets(config)

    if config["dataset"] == "blob":
        assert config["d"] == 2
        return generate_blob_datasets(config)

    if config["dataset"] in ["moon_and_blob", "blob_and_moon"]:
        assert config["d"] == 2
        return generate_moon_and_blob_datasets(config)

    if config["dataset"] == "mnist":
        return load_mnist()

    if config["dataset"] in ["MTPL2_frequency", "MTPL2_severity", "MTPL2_pure"]:
        return load_MTPL(config["dataset"][6:], config["n_dataset"], config["m"])

    raise NotImplementedError(f"The dataset '{config['dataset']}' is not supported.")
