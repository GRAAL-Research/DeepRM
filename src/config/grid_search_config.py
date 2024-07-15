import copy

from sklearn.model_selection import ParameterGrid

from src.config.utils import load_yaml_file_content, GRID_SEARCH_FILE_PATH, \
    get_sub_config_paths_keys, get_yaml_files_paths, get_yaml_files_merged_content


def create_config_combinations_sorted_by_dataset(config: dict) -> list[dict]:
    grid_search_config = load_yaml_file_content(GRID_SEARCH_FILE_PATH)

    datasets = [config["dataset"]]
    if "dataset" in grid_search_config:
        datasets = grid_search_config["dataset"]

    config_combinations = generate_combinations(config)
    return sort_config_combinations_by_dataset_name_idx(config_combinations, datasets)


def generate_combinations(config: dict) -> list[dict]:
    grid_search_config = load_yaml_file_content(GRID_SEARCH_FILE_PATH)

    overrode_config = overrode_config_with_grid_search_config(config, grid_search_config)
    parameter_grid = create_parameter_grid(overrode_config, grid_search_config)

    return [config_combination for config_combination in parameter_grid]


def sort_config_combinations_by_dataset_name_idx(config_combinations: list[dict], datasets: list[str]) -> list[dict]:
    return sorted(config_combinations,
                  key=lambda config_combination: datasets.index(config_combination["dataset"]))  # TODO


def validate_sub_config_paths_are_not_used_in_grid_search_config(grid_search_config: dict) -> None:
    sub_config_path_keys = get_sub_config_paths_keys(grid_search_config)

    assert_message = ("Using sub config paths in grid search is not yet supported. "
                      f"Remove {sub_config_path_keys} from {GRID_SEARCH_FILE_PATH}.")
    assert not sub_config_path_keys, assert_message


def overrode_config_with_grid_search_config(config: dict, grid_search_config: dict) -> dict:
    overrode_config = copy.deepcopy(config)
    validate_sub_config_paths_are_not_used_in_grid_search_config(grid_search_config)

    for param_name in grid_search_config:
        validate_parameter_name(param_name)
        overrode_config[param_name] = grid_search_config[param_name]

    return overrode_config


def create_parameter_grid(config: dict, grid_search_config: dict) -> ParameterGrid:
    config = {key: [value] if key not in grid_search_config else value for key, value in config.items()}
    return ParameterGrid([config])


def validate_parameter_name(parameter_name: str) -> None:
    yaml_files_paths = get_yaml_files_paths()
    yaml_files_paths.remove(GRID_SEARCH_FILE_PATH)

    if parameter_name not in get_yaml_files_merged_content(yaml_files_paths):
        error_message = f"You cannot use the parameter '{parameter_name}' because it's not in any config."
        raise KeyError(error_message)
