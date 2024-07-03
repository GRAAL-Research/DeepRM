from pathlib import Path

from git import Repo

from src.config.load import load_yaml_file_content


def create_config(config_name: str):
    config = load_yaml_file_content(Path("config") / config_name)

    if config["is_logging_commit_info"]:
        config = update_config_with_commit_name_and_hash(config)

    return config


def update_config_with_commit_name_and_hash(config: dict) -> dict:
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
