from pathlib import Path

import pandas as pd
import wandb

from src.utils.utils import TEST_ACCURACY_LABEL, TRAIN_ACCURACY_LABEL, VALID_ACCURACY_LABEL

CACHE_BASE_DIR = Path(__file__).parent / "wandb_cache"
CACHE_FILE_EXTENSION = "parquet"


def get_and_save_wandb_data(team: str, project: str):
    api = wandb.Api()
    runs = api.runs(f"{team}/{project}")

    runs_data = []
    for run in runs:
        if len(dict(run.summary)) <= 1:
            continue
        run_data = {
            TEST_ACCURACY_LABEL: run.summary[TEST_ACCURACY_LABEL],
            TRAIN_ACCURACY_LABEL: run.summary[TRAIN_ACCURACY_LABEL],
            VALID_ACCURACY_LABEL: run.summary[VALID_ACCURACY_LABEL],
            **run.config
        }
        runs_data.append(run_data)

    runs_data_frame = pd.DataFrame(runs_data)
    save_data_to_cache(runs_data_frame, team, project)

    return runs_data_frame


def save_data_to_cache(data: pd.DataFrame, team, project):
    if not CACHE_BASE_DIR.exists():
        CACHE_BASE_DIR.mkdir()
    data.to_parquet(CACHE_BASE_DIR / f"{team}_{project}.{CACHE_FILE_EXTENSION}")


def get_cached_data(team: str, project: str) -> None | pd.DataFrame:
    if Path(CACHE_BASE_DIR / f"{team}_{project}.{CACHE_FILE_EXTENSION}").exists():
        return pd.read_parquet(CACHE_BASE_DIR / f"{team}_{project}.{CACHE_FILE_EXTENSION}")

    return None


def fetch_wandb_data(team: str, project: str):
    cached_data = get_cached_data(team, project)
    if cached_data is not None:
        return cached_data

    return get_and_save_wandb_data(team, project)
