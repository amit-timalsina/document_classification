from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.common.utils.create_enum import create_enum_for_document_types
from document_classification.llm.prompt_technique import PromptTechnique, prompt_technique_to_model

if TYPE_CHECKING:
    from enum import Enum

    from pydantic import BaseModel

    from document_classification.llm.schemas.classification_entity import ClassificationEntity


def build_classification_schema(
    classifications: list[ClassificationEntity],
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
    DocumentType: type[Enum] = create_enum_for_document_types(  # noqa: N806
        enum_names=labels,
    )
    # Update the classification model with the correct DocumentType
    model_class.model_fields["classification"].annotation = DocumentType

    # Rebuild the model
    model_class.model_rebuild()

    return model_class
