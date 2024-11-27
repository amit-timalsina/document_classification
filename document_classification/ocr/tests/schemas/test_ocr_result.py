import pandas as pd
import pytest

from document_classification.ocr.config import ocr_config
from document_classification.ocr.schemas.ocr_result import OcrResult


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "text": ["Hello", "World"],
            "confidence": [0.9, 0.8],
            "page": [1, 1],
            "block": [1, 2],
            "paragraph": [1, 1],
            "line": [1, 2],
            "word_num": [1, 2],
            "x0": [10.0, 20.0],
            "y0": [30.0, 40.0],
            "x2": [15.0, 25.0],
            "y2": [35.0, 45.0],
            "space_type": ["word", "word"],
            "index_sort": [0, 1],
        },
    )


class TestOcrResult:
    """Test suite for the OcrResult class."""

    def test_ocr_result_init(self):
        """Test OcrResult initialization with no data."""
        result = OcrResult()
        assert isinstance(result.ocr_df, pd.DataFrame)
        assert result.ocr_df.empty

    def test_ocr_result_with_data(self, sample_df: pd.DataFrame):
        """Test OcrResult initialization with data."""
        result = OcrResult(ocr_df=sample_df)
        assert not result.ocr_df.empty
        assert len(result.ocr_df) == len(sample_df)

    def test_standardize_output(self, sample_df: pd.DataFrame):
        """Test standardization of OCR results."""
        result = OcrResult(ocr_df=sample_df)
        standardized = result.standardize_output()
        assert all(col in standardized.columns for col in ocr_config.output_columns)
        assert len(standardized.columns) == len(ocr_config.output_columns)

    def test_ocr_text_empty(self):
        """Test OCR text when dataframe is empty."""
        result = OcrResult()
        assert result.ocr_text == ""

    def test_ocr_text_with_data(self, sample_df: pd.DataFrame):
        """Test OCR text when dataframe has data."""
        result = OcrResult(ocr_df=sample_df)
        assert "Hello" in result.ocr_text
        assert "World" in result.ocr_text

    def test_ocr_dict_empty(self):
        """Test OCR dictionary when dataframe is empty."""
        result = OcrResult()
        assert result.ocr_dict == []

    def test_ocr_dict_with_data(self, sample_df: pd.DataFrame):
        """Test OCR dictionary when dataframe has data."""
        result = OcrResult(ocr_df=sample_df)
        dict_result = result.ocr_dict
        assert len(dict_result) == sample_df.shape[0]
        assert dict_result[0]["text"] == "Hello"
        assert dict_result[1]["text"] == "World"
        assert "space_type" in dict_result[0]
        assert "index_sort" in dict_result[0]
