from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from common.utils.file_utils import json_to_dataframe, load_json_file
from language_model.ocr_dataset import BoundingBox, OCRDataset, OCRwithBBoxDataset
from logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def custom_collate_fn(batch):
    texts, labels, bboxes = zip(*batch)

    # Pad the bounding boxes to the same length
    max_bbox_len = max(len(bbox) for bbox in bboxes)
    padded_bboxes = [
        bbox + [BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)] * (max_bbox_len - len(bbox))
        for bbox in bboxes
    ]

    # Convert bounding boxes to tensor
    bbox_tensors = [
        torch.tensor([[b.x_min, b.y_min, b.x_max, b.y_max] for b in bbox]) for bbox in padded_bboxes
    ]
    bbox_tensors = pad_sequence(bbox_tensors, batch_first=True)

    return {
        "texts": texts,
        "labels": torch.tensor(labels),
        "bboxes": bbox_tensors,
    }


class BaseDatasetPreparer(ABC):
    @abstractmethod
    def process_data(self, ocr_json_path: Path) -> tuple[list[Any], list[str]]:
        pass

    @abstractmethod
    def create_dataset(self, data: Any, labels: list[int]) -> Dataset:
        pass

    def prepare_data(
        self,
        ocr_json_path: Path,
        batch_size: int,
        processed_data_path: Path,
    ) -> tuple[DataLoader, DataLoader, LabelEncoder]:
        if processed_data_path.exists():
            logger.info("Loading preprocessed data...")
            with processed_data_path.open("rb") as f:
                train_dataset, val_dataset, label_encoder = pickle.load(f)
            logger.warning("Using pickle to load data. Ensure the source is trusted.")
        else:
            logger.info("Processing OCR data...")
            data, labels = self.process_data(ocr_json_path)

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

            train_data, val_data, train_labels, val_labels = train_test_split(
                data,
                encoded_labels,
                test_size=0.2,
                random_state=42,
            )

            train_dataset = self.create_dataset(train_data, train_labels)
            val_dataset = self.create_dataset(val_data, val_labels)
            logger.info("Saving preprocessed data...")
            with processed_data_path.open("wb") as f:
                pickle.dump((train_dataset, val_dataset, label_encoder), f)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
        )

        return train_dataloader, val_dataloader, label_encoder


class TextOnlyDatasetPreparer(BaseDatasetPreparer):
    def process_data(self, ocr_json_path: Path) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        labels: list[str] = []

        for label_folder_path in ocr_json_path.iterdir():
            if label_folder_path.is_dir():
                label = label_folder_path.name
                for file_path in label_folder_path.iterdir():
                    if file_path.is_file():
                        json_data = load_json_file(file_path)
                        ocr_df = json_to_dataframe(json_data)
                        ocr_text = " ".join(ocr_df["text"])
                        texts.append(ocr_text)
                        labels.append(label)
        return texts, labels

    def create_dataset(self, data: list[str], labels: list[int]) -> Dataset:
        return OCRDataset(data, labels)


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
