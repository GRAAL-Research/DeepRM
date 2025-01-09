import time

import numpy as np

from src.utils.epoch_logger import EpochLogger
from src.utils.utils import (Metric, TRAIN_ACCURACY_MEAN, VALID_ACCURACY_MEAN, TEST_ACCURACY_MEAN, TRAIN_LOSS,
                             VALID_LOSS, TEST_LOSS)


def log_epoch_info_in_terminal(config: dict, train_metrics: dict[str, np.ndarray], valid_metrics: dict[str, np.ndarray],
                               test_metrics: dict[str, np.ndarray], train_var: np.ndarray,
                               valid_var: np.ndarray, test_var: np.ndarray, start_time: float, epoch_idx: int,
                               are_test_bounds_computed: bool) -> None:
    formated_epoch_idx = "0" * (epoch_idx + 1 < 100) + "0" * (epoch_idx + 1 < 10) + str(epoch_idx + 1)

    if are_test_bounds_computed:
        bound_info_to_log = (
            f" - bounds: (lin: {test_metrics[Metric.LINEAR_BOUND_MEAN.value]:.2f}), "
            f"(hyp: {test_metrics[Metric.HPARAM_BOUND_MEAN.value]:.2f}), "
            f"(kl: {test_metrics[Metric.KL_BOUND_MEAN.value]:.2f}), "
            f"(kl_disintegrated: {test_metrics[Metric.KL_DISINTEGRATED_BOUND_MEAN.value]:.2f}), "
            f"(marchand: {test_metrics[Metric.MARCHAND_BOUND_MEAN.value]:.2f})")
    else:
        bound_info_to_log = ""

    time_info_to_log = f" - time: {round(time.time() - start_time)}s"

    if config["task"] == "classification":
        EpochLogger.log(
            f"epoch {formated_epoch_idx} - train_acc: {train_metrics[TRAIN_ACCURACY_MEAN]:.3f} - val_acc: {valid_metrics[VALID_ACCURACY_MEAN]:.3f}"
            f" - {TEST_ACCURACY_MEAN}: {test_metrics[TEST_ACCURACY_MEAN]:.3f}"
            f"{time_info_to_log}{bound_info_to_log}")
    elif config["task"] == "regression":
        EpochLogger.log(
            f"Epoch {formated_epoch_idx} - Train R2: {1 - train_metrics[TRAIN_LOSS] / train_var:.4f} - Val R2: {1 - valid_metrics[VALID_LOSS] / valid_var:.4f}"
            f" - Test R2: {1 - test_metrics[TEST_LOSS] / test_var:.4f}"
            f"{time_info_to_log}{bound_info_to_log}")
    else:
        raise NotImplementedError(f"The task '{config['task']}' is not supported.")
