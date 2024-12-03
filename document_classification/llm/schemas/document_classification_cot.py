from pydantic import Field

from document_classification.llm.schemas.document_classification import DocumentClassification


class DocumentClassificationCOT(DocumentClassification):
    """Schema for document classification using a chain of thought."""

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the classification",
    )
