import pandas as pd
import pytest

from document_classification.ocr.config import ocr_config


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "text": ["Hello", "World", "Foo", "Bar"],
            "confidence": [0.9, 0.8, 0.9, 0.8],
            "page": [1, 1, 1, 1],
            "block": [1, 1, 2, 2],
            "paragraph": [1, 1, 1, 1],
            "line": [1, 1, 2, 2],
            "word_num": [1, 2, 3, 4],
            "x0": [0.0, 6.0, 0.0, 6.0],
            "y0": [0.0, 0.0, 10.0, 10.0],
            "x2": [5.0, 10.0, 5.0, 10.0],
            "y2": [10.0, 10.0, 20.0, 20.0],
            "space_type": [1, 3, 2, 1],
            "index_sort": [0, 1, 2, 3],
        },
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame(
        columns=ocr_config.output_columns,
    )
