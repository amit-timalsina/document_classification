from __future__ import annotations

import pytest

from document_classification.common.schemas.line import Line
from document_classification.common.schemas.word import Word


@pytest.fixture
def words():
    """Fixture of words for Line schema tests."""
    return [
        Word(text="Hello", x0=0, y0=0, x2=5, y2=10),
        Word(text="World", x0=6, y0=0, x2=10, y2=10),
    ]


class TestLine:
    """Test suite for the Line schema."""

    def test_line_init(self, words: list[Word]):
        """Test that Line is initialized correctly."""
        line = Line(words=words)
        assert line.words == words

    def test_line_empty_init(self):
        """Test that Line can be initialized with an empty list of words."""
        line = Line(words=[])
        assert line.words == []

    def test_line_words_optional(self):
        """Test that Line can be initialized with an optional list of words."""
        line = Line()
        assert line.words == []
