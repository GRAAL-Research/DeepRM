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
    ACCURACY_MEAN = "acc"
    ACCURACY_STD = "std"
    LOSS = "loss"

    VALID_LINEAR_BOUND_MEAN = "valid_bound_lin_mean"
    VALID_HPARAM_BOUND_MEAN = "valid_bound_hyp_mean"
    VALID_KL_BOUND_MEAN = "valid_bound_kl_mean"
    VALID_MARCHAND_BOUND_MEAN = "valid_bound_mrch_mean"
    VALID_KL_DISINTEGRATED_BOUND_MEAN = "valid_bound_disintegrated_kl_mean"

    VALID_LINEAR_BOUND_STD = "valid_bound_lin_std"
    VALID_HPARAM_BOUND_STD = "valid_bound_hyp_std"
    VALID_KL_BOUND_STD = "valid_bound_kl_std"
    VALID_MARCHAND_BOUND_STD = "valid_bound_mrch_std"
    VALID_KL_DISINTEGRATED_BOUND_STD = "valid_bound_disintegrated_kl_std"

    TEST_LINEAR_BOUND_MEAN = "test_bound_lin_mean"
    TEST_HPARAM_BOUND_MEAN = "test_bound_hyp_mean"
    TEST_KL_BOUND_MEAN = "test_bound_kl_mean"
    TEST_MARCHAND_BOUND_MEAN = "test_bound_mrch_mean"
    TEST_KL_DISINTEGRATED_BOUND_MEAN = "test_bound_disintegrated_kl_mean"

    TEST_LINEAR_BOUND_STD = "test_bound_lin_std"
    TEST_HPARAM_BOUND_STD = "test_bound_hyp_std"
    TEST_KL_BOUND_STD = "test_bound_kl_std"
    TEST_MARCHAND_BOUND_STD = "test_bound_mrch_std"
    TEST_KL_DISINTEGRATED_BOUND_STD = "test_bound_disintegrated_kl_std"


def get_metric_name(set_type: SetType, metric_name: Metric) -> str:
    return f"{set_type.value}_{metric_name.value}"


TRAIN_ACCURACY_MEAN = get_metric_name(SetType.TRAIN, Metric.ACCURACY_MEAN)
TRAIN_ACCURACY_STD = get_metric_name(SetType.TRAIN, Metric.ACCURACY_STD)
TRAIN_LOSS = get_metric_name(SetType.TRAIN, Metric.LOSS)
VALID_ACCURACY_MEAN = get_metric_name(SetType.VALID, Metric.ACCURACY_MEAN)
VALID_ACCURACY_STD = get_metric_name(SetType.VALID, Metric.ACCURACY_STD)
VALID_LOSS = get_metric_name(SetType.VALID, Metric.LOSS)
TEST_ACCURACY_MEAN = get_metric_name(SetType.TEST, Metric.ACCURACY_MEAN)
TEST_ACCURACY_STD = get_metric_name(SetType.TEST, Metric.ACCURACY_STD)
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
