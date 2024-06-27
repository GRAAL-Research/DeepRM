from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def get_config(config_name: str) -> dict[str, Any]:
    omega_config = OmegaConf.load(Path("config") / config_name)
    omega_grid_search_config = OmegaConf.load(Path("config") / "grid_search_override.yaml")
    config = OmegaConf.to_container(omega_config, resolve=True)
    grid_search_config = OmegaConf.to_container(omega_grid_search_config, resolve=True)

    for parameter in grid_search_config:
        if parameter in config:
            config[parameter] = grid_search_config[parameter]
        else:
            config[parameter] = [config[parameter]]

    return config
