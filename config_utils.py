import copy
from pathlib import Path

from git import Repo
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid


def create_config(config_name: str):
    config = load_yaml_file_content(Path("config") / config_name)

    if config["is_saving_commit_hash"]:
        config = update_config_with_commit_name_and_hash(config)

    return config


def create_config_combinations_sorted_by_dataset(config: dict) -> list[dict]:
    grid_search_config = load_yaml_file_content(Path("config") / "grid_search_override.yaml")

    datasets = [config["dataset"]]
    if "dataset" in grid_search_config:
        datasets = grid_search_config["dataset"]

    config_combinations = generate_combinations(config)
    return sort_config_combinations_by_dataset_name_idx(config_combinations, datasets)


def generate_combinations(config: dict) -> list[dict]:
    grid_search_config = load_yaml_file_content(Path("config") / "grid_search_override.yaml")

    overrode_config = overrode_config_with_grid_search_config(config, grid_search_config)
    parameter_grid = create_parameter_grid(overrode_config, grid_search_config)

    return [config_combination for config_combination in parameter_grid]


def sort_config_combinations_by_dataset_name_idx(config_combinations: list[dict], datasets: list[str]) -> list[dict]:
    return sorted(config_combinations,
                  key=lambda config_combination: datasets.index(config_combination["dataset"]))  # TODO


def overrode_config_with_grid_search_config(config: dict, grid_search_config: dict) -> dict:
    overrode_config = copy.deepcopy(config)

    for param_name in grid_search_config:
        if param_name in overrode_config:
            overrode_config[param_name] = grid_search_config[param_name]
        else:
            error_message = (f"You cannot override the parameter '{param_name}'"
                             "during grid search because it's not in the config.")
            raise KeyError(error_message)

    return overrode_config


def create_parameter_grid(config: dict, grid_search_config: dict) -> ParameterGrid:
    config = {key: [value] if key not in grid_search_config else value for key, value in config.items()}
    return ParameterGrid([config])


def load_yaml_file_content(file_path: Path) -> dict:
    omega_config = OmegaConf.load(file_path)
    return OmegaConf.to_container(omega_config, resolve=True)


def update_config_with_commit_name_and_hash(config: dict) -> dict:
    repo = Repo()

    if repo.is_dirty(untracked_files=True):
        unstaged_files_paths = repo.untracked_files + [diff.a_path for diff in repo.index.diff(None)]
        staged_files_paths = [diff.a_path for diff in repo.index.diff(repo.head.commit)]

        error_message = ("We want deterministic experiments and your working tree isn't clean."
                         f"Please commit your changes in {unstaged_files_paths + staged_files_paths}")
        raise Exception(error_message)

    config["commit_hash"] = get_commit_hash(repo)
    config["commit_name"] = get_commit_name(repo)

    return config


def get_commit_hash(repo: Repo) -> str:
    latest_commit = repo.head.commit
    return latest_commit.hexsha


def get_commit_name(repo: Repo) -> str:
    latest_commit = repo.head.commit
    return latest_commit.message
