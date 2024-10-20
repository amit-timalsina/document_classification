import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from common.utils.dataframe_to_text import df_to_text
from ocr.config import ocr_config


class OcrResult(BaseModel):
    """OCR result schema."""

    ocr_df: pd.DataFrame = Field(default_factory=lambda: pd.DataFrame())

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def standardize_output(self) -> pd.DataFrame:
        """Standardize the OCR response to the expected output format."""
        # return certain columns
        return self.ocr_df[ocr_config.output_columns]

    @property
    def ocr_text(self) -> str:
        """Return the OCR text."""
        return df_to_text(self.ocr_df)
