from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for language model."""

    num_labels: int
    max_length: int = 512
    model_name: str = "bert-base-uncased"
