from git import Repo

from src.config.utils import load_yaml_file_content, CONFIG_BASE_PATH, get_sub_config_paths_keys, CONFIG_PATH, \
    get_sub_config_paths_values


def create_hyperparameters_config():
    config = load_yaml_file_content(CONFIG_PATH)
    add_sub_config_parameters(config)
    if config["is_logging_commit_info"]:
        config = add_commit_name_and_hash_to_the_config(config)
    return config


def add_sub_config_parameters(config: dict) -> dict:
    for sub_config_paths_key in get_sub_config_paths_keys(config):
        sub_config_file_path = CONFIG_BASE_PATH / config[sub_config_paths_key]
        sub_config = load_yaml_file_content(sub_config_file_path)
        validate_keys_are_not_duplicated_across_config_files(config, sub_config)

        config |= sub_config

    return config


def validate_keys_are_not_duplicated_across_config_files(config: dict, sub_config: dict) -> None:
    keys_intersection = compute_keys_intersection(sub_config, config)
    possible_config_with_duplicated_keys = [CONFIG_PATH.name] + get_sub_config_paths_values(config)

    assertion_message = (f"{keys_intersection} are duplicated keys across config files: "
                         f"{possible_config_with_duplicated_keys}.")
    assert not keys_intersection, assertion_message


def compute_keys_intersection(a: dict, b: dict) -> list[str]:
    intersection = set(a) & set(b)
    return list(intersection)


def add_commit_name_and_hash_to_the_config(config: dict) -> dict:
    repo = Repo()

    if repo.is_dirty(untracked_files=True):
        unstaged_files_paths = repo.untracked_files + [diff.a_path for diff in repo.index.diff(None)]
        staged_files_paths = [diff.a_path for diff in repo.index.diff(repo.head.commit)]

        error_message = ("Experiments might not be reproducible if your working tree isn't clean."
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
