from pathlib import Path

import numpy as np

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH

NUMPY_FILE_EXTENSION = ".npy"
CACHE_DIRECTORY = "cache"


def create_cache_path(config: dict, cache_base_path: Path) -> Path:
    dataset_config_not_overrode_by_grid_search_config = load_yaml_file_content(
        CONFIG_BASE_PATH / config["dataset_config_path"])
    file_name = []
    for key in dataset_config_not_overrode_by_grid_search_config.keys():
        if key not in ["task", "target_size", "criterion", "is_dataset_balanced"]:
            file_name.append(f"{key}={config[key]}")

    return cache_base_path / ("-".join(file_name) + NUMPY_FILE_EXTENSION)


def store_datasets_in_cache(config: dict, cache_base_path: Path, datasets: np.ndarray) -> None:
    cache_base_path.mkdir(exist_ok=True)
    np.save(create_cache_path(config, cache_base_path), datasets)


def create_cache_base_path(data_base_path: Path) -> Path:
    return data_base_path / CACHE_DIRECTORY
