import random
from enum import Enum
from pathlib import Path

import numpy as np
import torch


class SetType(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Metric(Enum):
    ACCURACY = "acc"
    LOSS = "loss"
    LINEAR_BOUND = "bound_lin"
    HPARAM_BOUND = "bound_hyp"
    KL_BOUND = "bound_kl"
    MARCHAND_BOUND = "bound_mrch"


def get_metric_name(set_type: SetType, metric_name: Metric) -> str:
    return f"{set_type.value}_{metric_name.value}"


TRAIN_ACCURACY = get_metric_name(SetType.TRAIN, Metric.ACCURACY)
TRAIN_LOSS = get_metric_name(SetType.TRAIN, Metric.LOSS)
VALID_ACCURACY = get_metric_name(SetType.VALID, Metric.ACCURACY)
VALID_LOSS = get_metric_name(SetType.VALID, Metric.LOSS)
TEST_ACCURACY = get_metric_name(SetType.TEST, Metric.ACCURACY)
TEST_LOSS = get_metric_name(SetType.TEST, Metric.LOSS)

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
