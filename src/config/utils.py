from pathlib import Path

from omegaconf import OmegaConf

CONFIG_BASE_PATH = Path("config")

GRID_SEARCH_FILE_PATH = CONFIG_BASE_PATH / "grid_search_override.yaml"
CONFIG_PATH = CONFIG_BASE_PATH / "config.yaml"


def load_yaml_file_content(file_path: Path) -> dict:
    omega_config = OmegaConf.load(file_path)
    return OmegaConf.to_container(omega_config, resolve=True)


def get_yaml_files_merged_content(yaml_files_paths: list[Path]) -> dict:
    merged_configs = {}
    for path in yaml_files_paths:
        merged_configs |= load_yaml_file_content(path)

    return merged_configs


def get_yaml_files_paths() -> list[Path]:
    return list(CONFIG_BASE_PATH.rglob("*.yaml"))


def get_sub_config_paths_keys(config: dict) -> list[str]:
    return [key for key in config if is_key_a_config_paths_key(key)]


def get_sub_config_paths_values(config: dict) -> list[str]:
    return [value for key, value in config.items() if is_key_a_config_paths_key(key)]


def is_key_a_config_paths_key(key: str) -> bool:
    return key.endswith("_config_path")
