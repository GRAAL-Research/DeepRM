import numpy as np

from src.utils.utils import TRAIN_ACCURACY_MEAN


def is_stopping_training(config: dict, train_metrics: dict[str, np.ndarray], epoch_idx: int, best_epoch: int) -> bool:
    has_no_learning_being_made = train_metrics[TRAIN_ACCURACY_MEAN] < 0.525 and epoch_idx > 50
    has_no_improvement_for_a_while = epoch_idx - best_epoch > config["early_stopping_patience"]

    return has_no_learning_being_made or has_no_improvement_for_a_while
