from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langsmith import traceable
from pydantic import BaseModel

if TYPE_CHECKING:
    from instructor import AsyncInstructor


class BaseLLMClassifier(ABC):
    """
    A base class for LLM-based classifiers.

    This class provides a common interface for implementation based on llm providers.
    """

    def __init__(self, client: AsyncInstructor, llm_model: str) -> None:
        """Initialize the classifier."""
        self.client = client
        self.llm_model = llm_model
        self.sem = asyncio.Semaphore(5)

    @abstractmethod
    async def classify(
        self,
        text: str,
        classification_schema: type[BaseModel],
    ) -> tuple[str, BaseModel]:
        """Perform classification on the input text."""

    async def classify_documents(
        self,
        texts: list[str],
        classification_schema: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Classify a list of document texts asynchronously."""
        tasks = [self.classify(text, classification_schema) for text in texts]

        resps = []
        for task in asyncio.as_completed(tasks):
            text, label = await task
            resps.append(
                {
                    "input": text,
                    "classification": label.model_dump() if isinstance(label, BaseModel) else label,
                },
            )
        return resps


class OpenAILLMClassifier(BaseLLMClassifier):
    """
    A class for classifying documents using OpenAI's LLM.

    This classifier uses an AsyncInstructor client to interact with an LLM (like gpt-4o)
    for document classification tasks.

    Note: It supports different output structures for LLM response.
    """

    @traceable(name="classify-document")
    async def classify(
        self,
        text: str,
        classification_model: type[BaseModel],
    ) -> tuple[str, BaseModel]:
        """Perform classification on the input text."""
        async with self.sem:  # some simple rate limiting
            classification = await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=classification_model,
                max_retries=2,
                messages=[
                    {
                        "role": "user",
                        "content": f"Classify the following text: {text}",
                    },
                ],
                strict=False,
            )
            return text, classification
