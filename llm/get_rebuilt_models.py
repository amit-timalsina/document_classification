from __future__ import annotations

from typing import TYPE_CHECKING

from llm.config import OCR_JSON_DIRECTORY

from .document_type import create_document_type_enum

if TYPE_CHECKING:
    from enum import Enum

    from pydantic import BaseModel

    from llm.pydantic_models import Classifications


def get_rebuilt_models(
    classifications: list[Classifications],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Rebuild the BaseDocumentClassification and DocumentClassificationCOT models."""
    from .pydantic_models import BaseDocumentClassification, DocumentClassificationCOT

    # Update field description of classification filed
    description = "Classify the document into one of the following labels:\n"
    labels = []
    for classification in classifications:
        description += f"\t{classification.label}: {classification.description}\n"
        labels.append(classification.label)
    BaseDocumentClassification.model_fields["classification"].description = description
    DocumentClassificationCOT.model_fields["classification"].description = description

    # Initialize DocumentType
    DocumentType: type[Enum] = create_document_type_enum(  # noqa: N806
        OCR_JSON_DIRECTORY,
        subset=labels,
    )
    # Update the classification models with the correct DocumentType
    BaseDocumentClassification.model_fields["classification"].annotation = DocumentType
    DocumentClassificationCOT.model_fields["classification"].annotation = DocumentType

    # Rebuild the models
    BaseDocumentClassification.model_rebuild()
    DocumentClassificationCOT.model_rebuild()

    return BaseDocumentClassification, DocumentClassificationCOT
