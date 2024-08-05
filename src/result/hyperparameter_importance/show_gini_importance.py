import pandas as pd

from src.result.hyperparameter_importance.utils import (plot_feature_importance, get_processed_x_and_y,
                                                        get_trained_random_forest)


def show_gini_importance(data: pd.DataFrame) -> None:
    x, y = get_processed_x_and_y(data)
    model = get_trained_random_forest(x, y)

    feature_names = list(x.columns)
    plot_feature_importance(feature_names, model.feature_importances_, "Gini Importance", "Gini Importance")
