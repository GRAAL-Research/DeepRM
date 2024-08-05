import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from src.result.hyperparameter_importance.shap_values_cache import compute_shap_values
from src.result.hyperparameter_importance.utils import get_processed_x_and_y, get_trained_random_forest
from src.utils.utils import FIGURE_BASE_PATH


def show_shap_values(data: pd.DataFrame, team: str, project: str, n_instances: int = 100) -> None:
    x, y = get_processed_x_and_y(data)
    model = get_trained_random_forest(x, y)
    explainer = shap.TreeExplainer(model)

    shap_values = compute_shap_values(x, team, project, explainer, n_instances)
    x = x[:n_instances]

    show_and_save_shap_plot(shap_values, x, n_instances, plot_type="dot", alpha=0.3)
    show_and_save_shap_plot(shap_values, x, n_instances, plot_type="bar")


def show_and_save_shap_plot(shap_values: np.ndarray, x: np.ndarray, n_instances: int, plot_type: str,
                            alpha: float = 1.0) -> None:
    shap_figure_path = FIGURE_BASE_PATH / "shap"
    if not shap_figure_path.exists():
        shap_figure_path.mkdir(parents=True)

    shap.summary_plot(shap_values, x, alpha=alpha, plot_type=plot_type, show=False)
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(shap_figure_path / f"{plot_type}-plot-{n_instances}.png")
    plt.show()
    plt.clf()
