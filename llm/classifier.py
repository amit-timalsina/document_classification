from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langsmith import traceable
from pydantic import BaseModel

if TYPE_CHECKING:
    from instructor import AsyncInstructor


class LLMClassifier:
    """
    A class for classifying documents using LLM.

    This classifier uses an AsyncInstructor client to interact with an LLM (like gpt-4o)
    for document classification tasks. It supports different output structures for LLM response.
    """

    def __init__(self, client: AsyncInstructor, llm_model: str = "gpt-4o") -> None:
        """Initialize the classifier."""
        self.client = client
        self.llm_model = llm_model
        self.sem = asyncio.Semaphore(5)

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

    async def classify_documents(
        self,
        texts: list[str],
        classification_model: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Classify a list of document texts asynchronously."""
        tasks = [self.classify(text, classification_model) for text in texts]

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
