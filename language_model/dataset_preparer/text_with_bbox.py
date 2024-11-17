from __future__ import annotations

from typing import TYPE_CHECKING

from common.utils.file_utils import json_to_dataframe, load_json_file
from language_model.dataset_preparer.base import BaseDatasetPreparer
from language_model.ocr_dataset import BoundingBox, OCRwithBBoxDataset

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from torch.utils.data import Dataset


class TextWithBBoxDatasetPreparer(BaseDatasetPreparer):
    def process_data(
        self,
        ocr_json_path: Path,
    ) -> tuple[list[tuple[str, list[list[float]]]], list[str]]:
        data: list[tuple[str, list[list[float]]]] = []
        labels: list[str] = []

        for label_folder_path in ocr_json_path.iterdir():
            if label_folder_path.is_dir():
                label: str = label_folder_path.name
                for file_path in label_folder_path.iterdir():
                    if file_path.is_file():
                        json_data: dict = load_json_file(file_path)
                        ocr_df: pd.DataFrame = json_to_dataframe(json_data)
                        ocr_text: str = " ".join(ocr_df["text"])
                        bboxes: list[list[float]] = (
                            ocr_df[["x0", "y0", "x2", "y2"]].to_numpy().tolist()
                        )
                        data.append((ocr_text, bboxes))
                        labels.append(label)
        return data, labels

    def create_dataset(
        self,
        data: list[tuple[str, list[list[float]]]],
        labels: list[int],
    ) -> Dataset:
        texts: list[str] = [item[0] for item in data]
        bboxes: list[list[list[float]]] = [item[1] for item in data]

        bounding_boxes: list[list[BoundingBox]] = [
            [
                BoundingBox(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3])
                for bbox in sublist
            ]
            for sublist in bboxes
        ]
        return OCRwithBBoxDataset(texts, labels, bounding_boxes)
