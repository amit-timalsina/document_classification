import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ocr.config import ocr_config


class OCRResult(BaseModel):
    df: pd.DataFrame = Field(default_factory=lambda: pd.DataFrame())

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def standardize_output(self) -> pd.DataFrame:
        # return certain columns
        return self.df[ocr_config.output_columns]

    def get_text(self) -> str:
        return " ".join(self.df["text"])
