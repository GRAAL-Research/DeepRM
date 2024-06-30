import numpy as np

from dataset.blob import generate_random_blob_dataset
from dataset.datasets_utils import create_empty_datasets
from dataset.moon import generate_moon_dataset


def generate_moon_and_blob_datasets(config: dict) -> np.ndarray:
    datasets = create_empty_datasets(config)

    nb_of_choice = 1
    choices = ["blob", "moon"]
    randomly_selected_dataset = np.choose(nb_of_choice, choices)

    for dataset_idx in range(len(datasets)):
        if randomly_selected_dataset == "moon":
            datasets[dataset_idx] = generate_moon_dataset(config)
        elif randomly_selected_dataset == "blob":
            datasets[dataset_idx] = generate_random_blob_dataset(config)
        else:
            error_message = f"Can't generate '{randomly_selected_dataset}' dataset. Choose between {choices}."
            raise NotImplementedError(error_message)

    return datasets
