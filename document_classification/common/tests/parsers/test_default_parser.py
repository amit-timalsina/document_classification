from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from document_classification.common.parsers.config import parser_config
from document_classification.common.parsers.default_parser import DefaultParser

if TYPE_CHECKING:
    from document_classification.common.schemas import Word


@pytest.fixture
def default_parser() -> DefaultParser:
    """Fixture for DefaultParser."""
    return DefaultParser()


class TestDefaultParser:
    """Test suite for DefaultParser."""

    def test_init(self, default_parser: DefaultParser) -> None:
        """Test DefaultParser initialization."""
        assert (
            default_parser.merge_threshold == parser_config.merge_threshold
        ), f"Initial merge threshold should be {parser_config.merge_threshold}"

    def test_parse(self, default_parser: DefaultParser, sample_df: pd.DataFrame) -> None:
        """Test DefaultParser parsing."""
        document = default_parser.parse(sample_df)
        assert len(document.lines) == 2, "Should identify 2 lines in the document"
        assert document.lines[0].words[0].text == "Hello", "First word in line 1 should be 'Hello'"
        assert document.lines[0].words[1].text == "World", "Second word in line 1 should be 'World'"
        assert document.lines[1].words[0].text == "Foo", "First word in line 2 should be 'Foo'"
        assert document.lines[1].words[1].text == "Bar", "Second word in line 2 should be 'Bar'"

    def test_words_in_lines(self, default_parser: DefaultParser, words: list[Word]) -> None:
        """Test DefaultParser words to lines conversion indirectly through parse."""
        # Create a DataFrame from the words for testing
        test_df = pd.DataFrame([word.__dict__ for word in words])
        document = default_parser.parse(test_df)
        assert len(document.lines) == 2, "Should create 2 lines from the words"
        assert [word.text for word in document.lines[0].words] == [
            "Hello",
            "World",
        ], "First line words mismatch"
        assert [word.text for word in document.lines[1].words] == [
            "Foo",
            "Bar",
        ], "Second line words mismatch"

    def test_empty_dataframe(self, default_parser: DefaultParser, empty_df: pd.DataFrame) -> None:
        """Test handling of an empty DataFrame."""
        document = default_parser.parse(empty_df)
        assert len(document.lines) == 0, "Document should have no lines from an empty DataFrame"

    def test_df_to_words(self, default_parser: DefaultParser, sample_df: pd.DataFrame) -> None:
        """Test DefaultParser DataFrame to words conversion indirectly through parse."""
        document = default_parser.parse(sample_df)
        words = document.lines[0].words + document.lines[1].words
        assert len(words) == 4, "Should be 4 words parsed from the DataFrame"
        assert {word.text for word in words} == {
            "Hello",
            "World",
            "Foo",
            "Bar",
        }, "Words mismatch from parsed document"
