from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from document_classification.llm.schemas.document_classification import DocumentClassification
from document_classification.llm.schemas.document_classification_cot import (
    DocumentClassificationCOT,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


class PromptTechnique(str, Enum):
    """The prompt technique to use."""

    ONE_SHOT = "one-shot"
    # FEW_SHOT = "few-shot"  # noqa: ERA001
    COT = "cot"


prompt_technique_to_model: dict[PromptTechnique, type[BaseModel]] = {
    PromptTechnique.ONE_SHOT: DocumentClassification,
    # PromptTechnique.FEW_SHOT: DocumentClassification,
    PromptTechnique.COT: DocumentClassificationCOT,
}
