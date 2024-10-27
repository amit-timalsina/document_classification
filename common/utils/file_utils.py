import json
from pathlib import Path

import pandas as pd


def load_json_file(file_path: Path) -> dict:
    with file_path.open(encoding="utf-8") as f:
        return json.load(f)


def json_to_dataframe(json_data: dict) -> pd.DataFrame:
    return pd.DataFrame(json_data)
