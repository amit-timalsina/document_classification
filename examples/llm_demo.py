"""
Demo of using LLMs for document classification.

Approaches to llm based extraction:

- one-shot ✅
- one-shot-COT ✅
- few-shot
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import instructor
import pandas as pd
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from document_classification.common.parsers.default_parser import DefaultParser
from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.utils.json_to_ocr_text import json_to_ocr_text
from document_classification.common.utils.parse_and_format import parse_and_format
from document_classification.llm.build_classification_schema import (
    build_classification_schema,
)
from document_classification.llm.classifier import OpenAILLMClassifier
from document_classification.llm.config import OCR_JSON_DIRECTORY
from document_classification.llm.evaluation import run_evaluation
from document_classification.llm.prompt_technique import PromptTechnique
from document_classification.llm.schemas.classification_entity import ClassificationEntity
from document_classification.logger import logger
from document_classification.ocr.providers.closed_source.google_vision.ocr import GoogleVisionOCR
from document_classification.ocr.readers.file_reader import FileReader

DEFAULT_CLASSIFICATIONS = [
    ClassificationEntity(
        label="form_1040",
        description="Form 1040 is the U.S. individual income tax return used by taxpayers to "
        "report personal income and calculate taxes owed",
    ),
    ClassificationEntity(
        label="form_1040_schedule_c",
        description="Form 1040 Schedule C is specifically used by sole proprietors to "
        "report income and expenses from a business.",
    ),
]


def create_openai_client() -> instructor.AsyncInstructor:
    """Create and configure OpenAI client with LangSmith wrapper."""
    oai_client = wrap_openai(AsyncOpenAI())
    return instructor.from_openai(oai_client)


def create_classifier(client: instructor.AsyncInstructor) -> OpenAILLMClassifier:
    """Create LLM classifier instance."""
    return OpenAILLMClassifier(client, llm_model="gpt-4o-2024-08-06")


async def predict_document_type_from_ocr_json(
    ocr_json_file_path: Path,
    *,
    prompt_technique: PromptTechnique,
) -> None:
    """Run document type prediction on a single file."""
    classification_schema = build_classification_schema(
        classifications=DEFAULT_CLASSIFICATIONS,
        prompt_technique=prompt_technique,
    )
    parser = DefaultParser()
    formatter = LayoutPreservingFormatter()
    ocr_text = json_to_ocr_text(ocr_json_file_path, parser, formatter)
    classifier = create_classifier(create_openai_client())
    classification_results = await classifier.classify_documents([ocr_text], classification_schema)
    logger.info(f"Inference results: {classification_results}")


async def predict_document_type(file_path: Path, *, prompt_technique: PromptTechnique) -> None:
    """Run document type prediction on a single file."""
    classification_schema = build_classification_schema(
        classifications=DEFAULT_CLASSIFICATIONS,
        prompt_technique=prompt_technique,
    )
    images = FileReader.read_file_from_path(file_path)
    ocr_provider = GoogleVisionOCR()
    ocr_dfs: list[pd.DataFrame] = []
    for image in images:
        ocr_result = ocr_provider.perform_ocr(image)
        ocr_dfs.append(ocr_result.ocr_df)

    ocr_df = pd.concat(ocr_dfs)
    parser = DefaultParser()
    formatter = LayoutPreservingFormatter()
    ocr_text = parse_and_format(ocr_df, parser, formatter)
    classifier = create_classifier(create_openai_client())
    classification_results = await classifier.classify_documents([ocr_text], classification_schema)
    logger.info(f"Inference results: {classification_results}")


async def evaluate_documents(
    directory: Path = OCR_JSON_DIRECTORY,
    *,
    prompt_technique: PromptTechnique,
) -> None:
    """Run evaluation on all documents in directory."""
    classification_model = build_classification_schema(
        classifications=DEFAULT_CLASSIFICATIONS,
        prompt_technique=prompt_technique,
    )
    evaluation_results, report = await run_evaluation(
        classifier=create_classifier(create_openai_client()),
        directory=directory,
        classification_model=classification_model,
        save_report_path=Path("llm_evaluation_report.json"),
    )
    log_evaluation_results(evaluation_results)
    logger.info(f"Report: {report}")


def log_evaluation_results(evaluation_results: dict) -> None:
    """Log evaluation results in a structured format."""
    for doc_type, results in evaluation_results.items():
        logger.info(f"Results for {doc_type}:")
        for result in results:
            logger.info(f"  File: {result['file']}")
            logger.info(f"  Classification: {result['classification']}")


if __name__ == "__main__":
    asyncio.run(
        predict_document_type(
            file_path=Path("files/e9e2ac9325664b4c9ca0324d5a5d782e.pdf"),
            prompt_technique=PromptTechnique.COT,
        ),
    )

    asyncio.run(
        predict_document_type_from_ocr_json(
            ocr_json_file_path=Path(
                "ocr_jsons_tesseract/form_1040_schedule_c/0fe2cd862cd64e85af6f98464b69e57b3.json",
            ),
            prompt_technique=PromptTechnique.COT,
        ),
    )

    asyncio.run(evaluate_documents(prompt_technique=PromptTechnique.COT))
