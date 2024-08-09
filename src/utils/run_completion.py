import json
from pathlib import Path

SEPARATOR = "\t"
CACHE_PATH = Path(__file__).parent / "completed_runs_cache"

print(CACHE_PATH)
def is_run_already_completed(config: dict) -> bool:
    file_path = CACHE_PATH / f"{config['project_name']}.json"
    if file_path.exists():
        configs = load_json_content(file_path)
        if config in configs:
            return True

    return False


def label_run_config_as_completed(config: dict) -> None:
    if not CACHE_PATH.exists():
        CACHE_PATH.mkdir()

    file_path = CACHE_PATH / f"{config['project_name']}.json"

    configs = load_json_content(file_path)
    configs.append(config)
    store_json_content(file_path, configs)


def load_json_content(file_path: Path) -> list[dict]:
    if not file_path.exists():
        store_json_content(file_path, [])

    with open(file_path, "r") as file:
        return json.load(file)


def store_json_content(file_path: Path, config: list[dict]) -> None:
    with open(file_path, "w") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)
