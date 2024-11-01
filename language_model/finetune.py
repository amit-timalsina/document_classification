from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from transformers import (  # type: ignore[import-untyped]
    BertForSequenceClassification,
    BertTokenizer,
)

from common.parsers.default_parser import DefaultParser
from common.parsers.layout_preserving_formatter import LayoutPreservingFormatter
from common.parsers.parse_and_format import parse_and_format
from common.utils.file_utils import json_to_dataframe, load_json_file
from logger import logger

if TYPE_CHECKING:
    from pathlib import Path


class OCRDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def process_ocr_file(
    file_path: Path,
    parser: DefaultParser,
    formatter: LayoutPreservingFormatter,
) -> str:
    json_data = load_json_file(file_path)
    ocr_df = json_to_dataframe(json_data)
    return parse_and_format(ocr_df, parser, formatter)


def process_ocr_data(ocr_json_path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    parser = DefaultParser()
    formatter = LayoutPreservingFormatter()

    for label_folder_path in ocr_json_path.iterdir():
        if label_folder_path.is_dir():
            label = label_folder_path.name
            for file_path in label_folder_path.iterdir():
                if file_path.is_file():
                    ocr_text = process_ocr_file(file_path, parser, formatter)
                    texts.append(ocr_text)
                    labels.append(label)
    return texts, labels


def prepare_data(ocr_json_path: Path, batch_size: int) -> tuple[DataLoader, LabelEncoder]:
    texts, labels = process_ocr_data(ocr_json_path)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    dataset = OCRDataset(texts, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, label_encoder


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_model(num_labels: int) -> tuple[BertForSequenceClassification, BertTokenizer]:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def train_model(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    tokenizer: BertTokenizer,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for batch_texts, batch_labels in dataloader:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = batch_labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.debug(f"Epoch {epoch+1}/{num_epochs} completed")


def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    save_path: Path,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path / "model")
    tokenizer.save_pretrained(save_path / "tokenizer")
