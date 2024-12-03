import pandas as pd

from document_classification.common.utils.dataframe_to_text import df_to_text


def test_df_to_text(sample_df: pd.DataFrame) -> None:
    """Test converting dataframe to plain text conserving blocks and lines."""
    text = df_to_text(sample_df)
    assert text == "Hello World\nFoo Bar"


def test_df_to_text_empty_df(empty_df: pd.DataFrame) -> None:
    """Test converting empty dataframe to plain text."""
    text = df_to_text(empty_df)
    assert text == ""


def test_df_to_text_null_space_type(sample_df: pd.DataFrame) -> None:
    # TODO (Amit): This test is not contributing to the coverage as None is converted to Nan.
    """Test converting dataframe with null space_type to plain text."""
    sample_df["space_type"] = [1, 1, 2, None]
    text = df_to_text(sample_df)
    assert text == "Hello World Foo Bar"


def test_df_to_text_higher_space_type(sample_df: pd.DataFrame) -> None:
    """Test converting dataframe with higher space_type to plain text."""
    sample_df["space_type"] = [1, 5, 2, 1]
    text = df_to_text(sample_df)
    assert text == "Hello World\n\nFoo Bar"


def test_df_to_text_0_space_type(sample_df: pd.DataFrame) -> None:
    """Test converting dataframe with 0 space_type to plain text."""
    sample_df["space_type"] = [0, 1, 2, 1]
    text = df_to_text(sample_df)
    assert text == "HelloWorld Foo Bar"
