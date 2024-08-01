import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.utils import TEST_ACCURACY_LABEL, TRAIN_ACCURACY_LABEL, VALID_ACCURACY_LABEL, FIGURE_BASE_PATH

COLOR_MAP = "PiYG"


def show_correlation_with_test_accuracy(data: pd.DataFrame, figure_file_name: str) -> None:
    important_hparams = find_important_hyperparameters(data)
    correlation_matrix = data[important_hparams].corr()[TEST_ACCURACY_LABEL].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    show_test_accuracy_values = True
    min_correlation_value = -1
    max_correlation_value = 1
    sns.heatmap(correlation_matrix.to_frame().T, annot=show_test_accuracy_values, vmin=min_correlation_value,
                vmax=max_correlation_value,
                cmap=COLOR_MAP, cbar_kws={"label": "Correlation"})
    plot_title = "Hyperparameters' Correlation with Test Accuracy"
    plt.title(plot_title)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    correlation_figure_path = FIGURE_BASE_PATH / "correlation"
    if not correlation_figure_path.exists():
        correlation_figure_path.mkdir(parents=True)
    plt.savefig(correlation_figure_path / f"{figure_file_name}.png")

    plt.show()


def find_important_hyperparameters(data: pd.DataFrame) -> list[str]:
    numerical_hparams = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numerical_hparams.remove(VALID_ACCURACY_LABEL)
    numerical_hparams.remove(TRAIN_ACCURACY_LABEL)

    correlation_matrix = data[numerical_hparams].corr()[TEST_ACCURACY_LABEL]

    important_numerical_params = []
    for numerical_param in numerical_hparams:
        if np.abs(correlation_matrix[numerical_param]) > 0:
            important_numerical_params.append(numerical_param)

    return important_numerical_params
