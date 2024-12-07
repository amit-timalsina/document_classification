from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langsmith import traceable
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

if TYPE_CHECKING:
    from instructor import AsyncInstructor
    from pydantic import BaseModel


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
    ) -> BaseModel:
        """Perform classification on the input text."""

    async def classify_documents(
        self,
        texts: list[str],
        classification_schema: type[BaseModel],
    ) -> list[BaseModel]:
        """Classify a list of document texts asynchronously."""
        tasks = [self.classify(text, classification_schema) for text in texts]

        predictions: list[BaseModel] = []
        for task in asyncio.as_completed(tasks):
            prediction = await task
            predictions.append(prediction)

        return predictions


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
    ) -> BaseModel:
        """Perform classification on the input text."""
        async with self.sem:  # some simple rate limiting
            return await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=classification_model,
                max_retries=2,
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are tasked with classifying a document given it's ocr text.",
                    ),
                    ChatCompletionUserMessageParam(role="user", content=text),
                ],
                strict=False,
            )
