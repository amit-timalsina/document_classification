from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from sklearn.metrics import classification_report  # type: ignore[import-untyped]

from document_classification.common.parsers.default_parser import DefaultParser
from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.utils.file_utils import json_to_dataframe, load_json_file
from document_classification.common.utils.parse_and_format import parse_and_format

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from document_classification.llm.classifier import OpenAILLMClassifier


async def run_evaluation(
    classifier: OpenAILLMClassifier,
    directory: Path,
    classification_model: type[BaseModel],
    report_format: Literal["json", "dataframe"] = "json",
    save_report_path: Path | None = None,
) -> tuple[dict, str | dict | pd.DataFrame]:
    """Run evaluation on all files in a directory."""
    results: dict[str, list[dict[str, Any]]] = {}
    all_true_labels = []
    all_predicted_labels = []

    for doc_type_dir in directory.iterdir():
        if doc_type_dir.is_dir():
            results[doc_type_dir.name] = []

            for json_file in doc_type_dir.glob("*.json"):
                json_data = load_json_file(json_file)
                ocr_df = json_to_dataframe(json_data)

                parser = DefaultParser()
                formatter = LayoutPreservingFormatter()
                ocr_text = parse_and_format(ocr_df, parser, formatter)

                predictions = await classifier.classify_documents(
                    [ocr_text],
                    classification_model,
                )
                predicted_label: str = predictions[0].classification.value  # type: ignore[attr-defined]
                true_label = doc_type_dir.name
                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_label)

                results[doc_type_dir.name].append(
                    {
                        "file": str(json_file),
                        "classification": predicted_label,
                    },
                )

    report = generate_report(all_true_labels, all_predicted_labels, report_format)
    if save_report_path is not None:
        save_report(report, save_report_path)

    return results, report


def save_report(
    report: str | dict | pd.DataFrame,
    path: Path,
) -> None:
    """Save the report to a file."""
    if isinstance(report, (str, dict)):
        with path.open("w") as f:
            json.dump(report, f, indent=4)
    elif isinstance(report, pd.DataFrame):
        report.to_csv(path, index=False)


def generate_report(
    true_labels: list[str],
    predicted_labels: list[str],
    report_format: Literal["json", "dataframe"] = "json",
) -> str | dict | pd.DataFrame:
    """Generate a classification report."""
    report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

    if report_format == "json":
        return report_dict
    if report_format == "dataframe":
        return pd.DataFrame(report_dict).transpose()
    msg = "Unsupported report format. Choose 'json' or 'csv'."
    raise ValueError(msg)
