import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.result.hyperparameter_importance.utils import plot_feature_importance, get_trained_random_forest, RANDOM_STATE, \
    get_processed_x_and_y


def show_permutation_importance(data: pd.DataFrame) -> None:
    x, y = get_processed_x_and_y(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    model = get_trained_random_forest(x_train, y_train)

    importance = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=RANDOM_STATE)

    feature_names = list(x.columns)
    plot_feature_importance(feature_names, importance.importances_mean, "Permutation Importance (mean)",
                            "Accuracy difference when shuffled")
    plot_feature_importance(feature_names, importance.importances_std, "Permutation Importance (std)",
                            "Accuracy difference when shuffled")
