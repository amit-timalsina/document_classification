from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from llm.config import OCR_JSON_DIRECTORY

from .document_type import create_document_type_enum
from .pydantic_models import BaseDocumentClassification, DocumentClassificationCOT

if TYPE_CHECKING:
    from pydantic import BaseModel

    from llm.pydantic_models import Classifications


class PromptTechnique(str, Enum):
    """The prompt technique to use."""

    ONE_SHOT = "one-shot"
    FEW_SHOT = "few-shot"
    COT = "cot"


prompt_technique_to_model: dict[PromptTechnique, type[BaseModel]] = {
    PromptTechnique.ONE_SHOT: BaseDocumentClassification,
    PromptTechnique.FEW_SHOT: BaseDocumentClassification,
    PromptTechnique.COT: DocumentClassificationCOT,
}


def get_rebuilt_model(
    classifications: list[Classifications],
    prompt_technique: PromptTechnique,
) -> type[BaseModel]:
    """Rebuild the classification model based on prompt technique."""
    description = "Classify the document into one of the following labels:\n"
    labels = []
    for classification in classifications:
        description += f"\t{classification.label}: {classification.description}\n"
        labels.append(classification.label)

    model_class = prompt_technique_to_model[prompt_technique]
    model_class.model_fields["classification"].description = description

    # Initialize DocumentType
    DocumentType: type[Enum] = create_document_type_enum(  # noqa: N806
        OCR_JSON_DIRECTORY,
        subset=labels,
    )
    # Update the classification model with the correct DocumentType
    model_class.model_fields["classification"].annotation = DocumentType

    # Rebuild the model
    model_class.model_rebuild()

    return model_class
