import multiprocessing
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.result.hyperparameter_importance.download_and_save_wandb_data import fetch_wandb_data
from src.utils.utils import TEST_ACCURACY_LABEL, FIGURE_BASE_PATH

ACCURACY_MIN_VALUE = 0
ACCURACY_MAX_VALUE = 1
COLOR_MAP = "viridis"


def show_hyperparameters_impact_on_test_accuracy(data: pd.DataFrame, hyperparameters: list[str]) -> None:
    data_filtered = data[hyperparameters + [TEST_ACCURACY_LABEL]]
    length_of_the_groups = 2
    hyperparameters_combinations = list(combinations(hyperparameters, length_of_the_groups))
    data_to_plot = [(data_filtered, x_hparam, y_hparam) for x_hparam, y_hparam in hyperparameters_combinations]

    with multiprocessing.Pool() as pool:
        pool.map(create_heatmap, data_to_plot)


def create_heatmap(data: pd.DataFrame) -> None:
    data, x_hparam, y_hparam = data
    mean_data = data.pivot_table(values=TEST_ACCURACY_LABEL, index=y_hparam, columns=x_hparam, aggfunc="mean")

    columns = mean_data.columns
    index = mean_data.index
    if len(index) == 1:
        logger.info(f"Skipping {x_hparam} vs {y_hparam} because there is only one value in {y_hparam}.")
        return
    if len(columns) == 1:
        logger.info(f"Skipping {x_hparam} vs {y_hparam} because there is only one value in {x_hparam}.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(mean_data, cmap=COLOR_MAP, vmin=ACCURACY_MIN_VALUE, vmax=ACCURACY_MAX_VALUE)
    ax.set_xlabel(x_hparam)
    ax.set_ylabel(y_hparam)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(index)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(index)

    for i in range(len(index)):
        for j in range(len(columns)):
            if mean_data.iloc[i, j] > 0.35:
                text_color = "black"
            else:
                text_color = "white"
            ax.text(j, i, f"{mean_data.iloc[i, j]:.2f}", ha="center", va="center", color=text_color)

    ax.set_title(f"Two Hyperparameters' impact on Test Accuracy")
    fig.colorbar(img, ax=ax, label="Test Accuracy")

    plt.tight_layout()

    hparam_impact_figure_path = FIGURE_BASE_PATH / "hparam_impact"
    if not hparam_impact_figure_path.exists():
        hparam_impact_figure_path.mkdir(parents=True)

    plt.savefig(hparam_impact_figure_path / f"{x_hparam}_and_{y_hparam}.png")
    plt.close(fig)


if __name__ == "__main__":
    team = "graal-deeprm2024"
    project = "message-module-with-kme-exp5-mnist"
    data = fetch_wandb_data(team, project)

    hyperparameters = ["lr", "batch_size", "msg_size", "tfm_input_dim", "tfm_n_heads", "tfm_mlp_dim", "tfm_n_encoders",
                       "tfm_drop_out", "tfm_output_dim"]
    show_hyperparameters_impact_on_test_accuracy(data, hyperparameters)
