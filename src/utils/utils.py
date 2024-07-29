import random

import numpy as np
import torch
from matplotlib import pyplot as plt

plt.switch_backend("agg")


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def create_run_name(config: dict) -> str:
    run_name_content = config["run_name_content"]

    if not run_name_content:
        return "Default name"

    run_name_elements = []
    for parameter_name in run_name_content:
        try:
            run_name_elements.append(f"{parameter_name.replace('_', '-')}={config[parameter_name]}")
        except KeyError:
            raise ValueError(f"Parameter '{parameter_name}' not found.")

    return "_".join(run_name_elements)


def validate_run_names(configs: list[dict]) -> None:
    for config in configs:
        create_run_name(config)
