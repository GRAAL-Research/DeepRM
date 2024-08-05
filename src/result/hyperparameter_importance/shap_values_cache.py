from pathlib import Path

import numpy as np
import pandas as pd
from shap import TreeExplainer

CACHE_BASE_DIR = Path(__file__).parent / "shap_cache"
FILE_EXTENSION = "npy"


def compute_shap_values(x: pd.DataFrame, project: str = None, team: str = None,
                        explainer: TreeExplainer = None, n_instances: int = 100) -> np.ndarray:
    x = x[:n_instances]
    expected_shap_cache_path = CACHE_BASE_DIR / f"{project}_{team}_{n_instances}-instances.{FILE_EXTENSION}"

    if expected_shap_cache_path.exists():
        return np.load(expected_shap_cache_path)

    return compute_and_store_shape_values(x, expected_shap_cache_path, explainer)


def compute_and_store_shape_values(x: np.ndarray, expected_datasets_cache_path: Path,
                                   explainer: TreeExplainer = None) -> np.ndarray:
    shape_values = explainer.shap_values(x)
    save_shap_values_in_cache(shape_values, expected_datasets_cache_path)
    return shape_values


def save_shap_values_in_cache(datasets: np.ndarray, expected_datasets_cache_path) -> None:
    CACHE_BASE_DIR.mkdir(exist_ok=True)
    np.save(expected_datasets_cache_path, datasets)
