"""
Demo of using LLMs for document classification.

Approaches to llm based extraction:

- one-shot ✅
- one-shot-COT ✅
- few-shot
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import instructor
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from llm.classifier import LLMClassifier
from llm.config import OCR_JSON_DIRECTORY
from llm.evaluation import run_evaluation
from llm.get_rebuilt_models import get_rebuilt_models
from llm.inference import run_inference
from llm.pydantic_models import Classifications
from logger import logger

if TYPE_CHECKING:
    from pydantic import BaseModel

# Wrap the OpenAI client with LangSmith
oai_client = wrap_openai(AsyncOpenAI())

# Patch the client with instructor
client = instructor.from_openai(oai_client)

classifier = LLMClassifier(client, llm_model="gpt-4o-2024-08-06")

default_classifications = classifications = [
    Classifications(
        label="form_1040",
        description="Form 1040 is the U.S. individual income tax return used by taxpayers to "
        "report personal income and calculate taxes owed",
    ),
    Classifications(
        label="form_1040_schedule_c",
        description="Form 1040 Schedule C is specifically used by sole proprietors to "
        "report income and expenses from a business.",
    ),
]


def get_classification_model(*, include_cot: bool) -> type[BaseModel]:
    """Get the appropriate classification model based on whether COT is included."""
    base_document_classification, base_document_classification_cot = get_rebuilt_models(
        classifications=default_classifications,
    )
    return base_document_classification_cot if include_cot else base_document_classification


async def main(*, predict_from_file: bool = True) -> None:
    """Run the demo."""
    classification_model = get_classification_model(include_cot=False)
    if predict_from_file:
        # Example of running inference on a single file
        file_path = OCR_JSON_DIRECTORY / "form_1040" / "3b8c728616344939b4c1a9056634263a.json"
        inference_result = await run_inference(
            classifier=classifier,
            file_path=file_path,
            classification_model=classification_model,
        )
        logger.info(f"Inference result: {inference_result}")

    else:
        # Example of running evaluation on all files
        evaluation_results = await run_evaluation(
            classifier=classifier,
            directory=OCR_JSON_DIRECTORY,
            classification_model=classification_model,
        )
        for doc_type, results in evaluation_results.items():
            logger.info(f"Results for {doc_type}:")
            for result in results:
                logger.info(f"  File: {result['file']}")
                logger.info(f"  Classification: {result['classification']}")


if __name__ == "__main__":
    asyncio.run(main(predict_from_file=False))
