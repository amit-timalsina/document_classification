from __future__ import annotations

import json
import random
import re
from pathlib import Path

from logger import logger


def mask_nums(x: str) -> str:
    """Preprocess the texts and masking numbers/amounts."""
    tmp = re.sub(r"\d", "X", x)
    tmp = re.sub(r"-", " ", tmp)
    tmp = re.sub(r"@", " ", tmp)
    tmp = re.sub(r":", " ", tmp)
    return re.sub(r"\/", " ", tmp)


def load_files_from_folder(folder_path: str) -> list[str]:
    """Load text files from a specific folder and return list of file contents."""
    files = []
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file():
            with file_path.open(encoding="utf-8") as file:
                data = json.load(file)
                if data:
                    # sort and convert to required format
                    data = sorted(data, key=lambda x: x["index_sort"])
                    text = " ".join([i["text"] for i in data])
                    text = re.sub(r"\s+", " ", text)
                    text = mask_nums(text)
                    files.append(text)
    return files


def split_data(
    data: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split data into training, validation, and test sets based on given ratios.

    Returns a tuple (train_data, val_data, test_data).
    """
    random.shuffle(data)
    total = len(data)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def write_to_file(data: list[str], file_path: str) -> None:
    """Write a list of data to a specified file."""
    with Path(file_path).open("w", encoding="utf-8") as file:
        for line in data:
            file.write(line + "\n")


def create_fasttext_dataset(data_folder_path: str, output_folder_path: str) -> None:
    """
    Create a fastText dataset from the given data folder.

    The data_folder contains sub folders for each document type.
    The sub folder names are used as labels. Each of these subfolders contains ocr json files
    for each document. Each ocr json file is a list of dictionaries where each dictionary contains
    `text` key.

    All text files in each sub folder are used to create train.txt, validation.txt, and
    test.txt in output_folder_path.
    Data format for each *.txt file is:

    __label__1 ocr_text
    __label__2 ocr_text
    __label__1 ocr_text
    ...

    """
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    all_train_data = []
    all_val_data = []
    all_test_data = []

    for label_folder_path in Path(data_folder_path).iterdir():
        if label_folder_path.is_dir():
            logger.debug(f"Processing label folder path: {label_folder_path}")
            files = load_files_from_folder(str(label_folder_path))
            logger.debug(f"Found {len(files)} files in folder: {label_folder_path}")
            labeled_data = [f"__label__{label_folder_path.name} {text}" for text in files]

            train_data, val_data, test_data = split_data(labeled_data)

            all_train_data.extend(train_data)
            all_val_data.extend(val_data)
            all_test_data.extend(test_data)

    # Write data to respective files
    write_to_file(all_train_data, str(Path(output_folder_path) / "train.txt"))
    write_to_file(all_val_data, str(Path(output_folder_path) / "validation.txt"))
    write_to_file(all_test_data, str(Path(output_folder_path) / "test.txt"))
