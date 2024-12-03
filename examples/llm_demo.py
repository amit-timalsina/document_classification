"""
Demo of using LLMs for document classification.

Approaches to llm based extraction:

- one-shot ✅
- one-shot-COT ✅
- few-shot
"""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TCH003

import instructor
import typer
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from document_classification.common.parsers.default_parser import DefaultParser
from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.utils.json_to_ocr_text import json_to_ocr_text
from document_classification.llm.classifier import OpenAILLMClassifier
from document_classification.llm.config import OCR_JSON_DIRECTORY
from document_classification.llm.evaluation import run_evaluation
from document_classification.llm.prompt_technique import PromptTechnique
from document_classification.llm.rebuild_classification_schema import (
    rebuild_classification_schema,
)
from document_classification.llm.schemas.classification_entity import ClassificationEntity
from document_classification.logger import logger

app = typer.Typer()

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


async def predict_document_type(file_path: Path, *, prompt_technique: PromptTechnique) -> None:
    """Run document type prediction on a single file."""
    classification_model = rebuild_classification_schema(
        classifications=DEFAULT_CLASSIFICATIONS,
        prompt_technique=prompt_technique,
    )
    parser = DefaultParser()
    formatter = LayoutPreservingFormatter()
    ocr_text = json_to_ocr_text(file_path, parser, formatter)
    classifier = create_classifier(create_openai_client())
    classification_results = await classifier.classify_documents([ocr_text], classification_model)
    logger.info(f"Inference results: {classification_results}")


async def evaluate_documents(
    directory: Path = OCR_JSON_DIRECTORY,
    *,
    prompt_technique: PromptTechnique,
) -> None:
    """Run evaluation on all documents in directory."""
    classification_model = rebuild_classification_schema(
        classifications=DEFAULT_CLASSIFICATIONS,
        prompt_technique=prompt_technique,
    )
    evaluation_results = await run_evaluation(
        classifier=create_classifier(create_openai_client()),
        directory=directory,
        classification_model=classification_model,
    )
    log_evaluation_results(evaluation_results)


def log_evaluation_results(evaluation_results: dict) -> None:
    """Log evaluation results in a structured format."""
    for doc_type, results in evaluation_results.items():
        logger.info(f"Results for {doc_type}:")
        for result in results:
            logger.info(f"  File: {result['file']}")
            logger.info(f"  Classification: {result['classification']}")


@app.command()
def inference(
    file_path: Path | None,
    *,
    prompt_technique: PromptTechnique = typer.Option(  # noqa: B008
        default=PromptTechnique.ONE_SHOT,
        help="Prompt technique to use",
    ),
) -> None:
    """Run prediction on a single file."""
    if file_path is None:
        msg = "File path is required"
        raise typer.BadParameter(msg)
    asyncio.run(predict_document_type(file_path, prompt_technique=prompt_technique))


@app.command()
def evaluate(
    directory: Path = OCR_JSON_DIRECTORY,
    *,
    prompt_technique: PromptTechnique = typer.Option(  # noqa: B008
        default=PromptTechnique.ONE_SHOT,
        help="Prompt technique to use",
    ),
) -> None:
    """Run evaluation on all files in directory."""
    asyncio.run(evaluate_documents(directory, prompt_technique=prompt_technique))


if __name__ == "__main__":
    app()
