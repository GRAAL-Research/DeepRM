from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from src.utils.utils import TEST_ACCURACY, VALID_ACCURACY, TRAIN_ACCURACY, FIGURE_BASE_PATH

RANDOM_STATE = 123


def plot_feature_importance(hyperparameters: list[str], importance: np.ndarray, title: str, y_label: str) -> None:
    hyperparameters, importance = create_sorted_and_filtered_importance(hyperparameters, importance)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(hyperparameters, importance)
    for bar in bars:
        height = bar.get_height()
        x_center_of_the_bar = bar.get_x() + bar.get_width() / 2
        plt.text(x_center_of_the_bar, height, f"{height:.3f}", ha="center", va="bottom")

    plt.title(f"{title} on Test Accuracy")
    plt.xlabel("Hyperparameter")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    importance_figure_path = FIGURE_BASE_PATH / "importance"
    if not importance_figure_path.exists():
        importance_figure_path.mkdir(parents=True)

    file_name = title.lower().replace(" ", "-")
    plt.savefig(importance_figure_path / f"{file_name}.png")

    plt.show()


def create_sorted_and_filtered_importance(feature_names, importances: np.ndarray, importance_threshold: float = 0.001):
    importance_data = pd.DataFrame({"feature_name": feature_names, "importance": importances})
    importance_data = importance_data[importance_data["importance"] >= importance_threshold]
    importance_data = importance_data.sort_values("importance", ascending=False)

    return importance_data["feature_name"], importance_data["importance"]


def get_processed_x_and_y(data: pd.DataFrame, is_dropping_na: bool = False) -> tuple:
    if is_dropping_na:
        data = data.dropna(axis="rows")
    else:
        if data.isna().any().any():
            logger.warning("The data contains NaN.")

    x = process_x(data)
    y = data[TEST_ACCURACY]

    return x, y


def get_trained_random_forest(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    return model


def process_x(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [TEST_ACCURACY,
                       VALID_ACCURACY,
                       TRAIN_ACCURACY,
                       "commit_hash",
                       "commit_name",
                       "run_name_content"
                       ]
    x = data.drop(columns_to_drop, axis="columns")
    x = x.applymap(stringify_list)
    x = pd.get_dummies(x, prefix_sep="=")
    return x.loc[:, x.nunique() != 1]


def stringify_list(item: Any) -> Any:
    if isinstance(item, list):
        return str(item)
    return item
