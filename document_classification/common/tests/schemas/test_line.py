from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.common.schemas.line import Line

if TYPE_CHECKING:
    from document_classification.common.schemas.word import Word


class TestLine:
    """Test suite for the Line schema."""

    def test_line_init(self, words: list[Word]):
        """Test that Line is initialized correctly."""
        line = Line(words=words)
        assert line.words == words

    def test_line_empty_init(self):
        """Test that Document can be initialized with an empty list of lines."""
        line = Line()
        assert line.words == []
