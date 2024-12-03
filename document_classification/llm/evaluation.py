from __future__ import annotations

from typing import TYPE_CHECKING, Any

from document_classification.common.parsers.default_parser import DefaultParser
from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.utils.file_utils import json_to_dataframe, load_json_file
from document_classification.common.utils.parse_and_format import parse_and_format

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from document_classification.llm.classifier import LLMClassifier


async def run_evaluation(
    classifier: LLMClassifier,
    directory: Path,
    classification_model: type[BaseModel],
) -> dict[str, list[dict[str, Any]]]:
    """Run evaluation on all files in a directory."""
    results: dict[str, list[dict[str, Any]]] = {}

    for doc_type_dir in directory.iterdir():
        if doc_type_dir.is_dir():
            results[doc_type_dir.name] = []

            for json_file in doc_type_dir.glob("*.json"):
                json_data = load_json_file(json_file)
                ocr_df = json_to_dataframe(json_data)

                parser = DefaultParser()
                formatter = LayoutPreservingFormatter()
                ocr_text = parse_and_format(ocr_df, parser, formatter)

                classification = await classifier.classify_documents(
                    [ocr_text],
                    classification_model,
                )
                results[doc_type_dir.name].append(
                    {
                        "file": str(json_file),
                        "classification": classification[0]["classification"],
                    },
                )

    return results
