import pandas as pd
import pytest


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
            "space_type": ["word", "word", "word", "word"],
            "index_sort": [0, 1, 2, 3],
        },
    )


@pytest.fixture
def empty_df():
    return pd.DataFrame(
        columns=[
            "text",
            "confidence",
            "page",
            "block",
            "paragraph",
            "line",
            "word_num",
            "x0",
            "y0",
            "x2",
            "y2",
            "space_type",
            "index_sort",
        ],
    )
