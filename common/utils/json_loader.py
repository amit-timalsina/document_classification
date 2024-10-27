from __future__ import annotations

import json
from pathlib import Path


class JsonLoader:
    """Loader for JSON files."""

    def __init__(self, json_dir: str) -> None:
        """
        Initialize the JSON loader.

        Args:
            json_dir: The directory containing JSON files.

        """
        self.json_dir = json_dir

    def load(self, file_id: str) -> dict:
        """
        Load a JSON file for a given file ID.

        Args:
            file_id: The ID/name of the file to load. Don't include extension as
                the class already knows it.

        Returns:
            dict: The loaded JSON data.

        """
        file_path = Path(self.json_dir) / f"{file_id}.json"
        with file_path.open(encoding="utf-8") as f:
            return json.load(f)

    def load_all(self) -> list[dict]:
        """
        Load all JSON files in the directory.

        Returns:
            list[dict]: A list of dictionaries, each representing a JSON file.

        """
        json_files = Path(self.json_dir).glob("*.json")
        return [json.load(file.open(encoding="utf-8", mode="r")) for file in json_files]
