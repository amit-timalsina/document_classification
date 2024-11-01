from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for BERT model training."""

    num_labels: int
    max_length: int = 512
    model_name: str = "bert-base-uncased"
