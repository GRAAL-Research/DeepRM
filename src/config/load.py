from pathlib import Path

from omegaconf import OmegaConf


def load_yaml_file_content(file_path: Path) -> dict:
    omega_config = OmegaConf.load(file_path)
    return OmegaConf.to_container(omega_config, resolve=True)


