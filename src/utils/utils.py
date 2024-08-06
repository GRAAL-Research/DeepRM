import random
from pathlib import Path

import numpy as np
import torch

TRAIN_ACCURACY = "train_acc"
TRAIN_LOSS = "train_loss"

VALID_ACCURACY = "valid_acc"
VALID_LOSS = "valid_loss"
epoch
TEST_ACCURACY = "test_acc"
TEST_LOSS = "test_loss"

LINEAR_BOUND = "bound_lin"
HYP_BOUND = "bound_hyp"
KL_BOUND = "bound_kl"
MARCHAND_BOUND = "bound_mrch"

FIGURE_BASE_PATH = Path(__file__).parent.parent.parent / "figures"


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
