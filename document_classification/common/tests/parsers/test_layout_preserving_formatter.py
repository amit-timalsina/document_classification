from document_classification.common.parsers.layout_preserving_formatter import (
    LayoutPreservingFormatter,
)
from document_classification.common.schemas import Document, Line, Word


class TestLayoutPreservingFormatter:
    """Test cases for the LayoutPreservingFormatter class."""

    def test_format_simple_document(self):
        """Test that the formatter can format a document with a single line of text."""
        formatter = LayoutPreservingFormatter()
        words = [
            Word(text="Hello", x0=0, y0=0, x2=50, y2=10),
            Word(text="World", x0=60, y0=0, x2=110, y2=10),
        ]
        line = Line(words=words)
        document = Document(lines=[line])
        formatted_text = formatter.format(document)

        assert (
            formatted_text == "Hello       World"
        ), "Formatted text should preserve word order and spacing"

    def test_format_with_long_line(self):
        """Test that the formatter can format a document with a long line of text."""
        formatter = LayoutPreservingFormatter(max_line_length=20)
        words = [
            Word(text="Hello", x0=0, y0=0, x2=50, y2=10),
            Word(text="World", x0=200, y0=0, x2=250, y2=10),
        ]
        line = Line(words=words)
        document = Document(lines=[line])
        formatted_text = formatter.format(document)

        assert (
            formatted_text == "Hello               World"
        ), "Should handle long lines with appropriate spacing"

    def test_format_empty_document(self):
        """Test that the formatter can format an empty document."""
        formatter = LayoutPreservingFormatter()
        document = Document(lines=[])
        formatted_text = formatter.format(document)

        assert formatted_text == "", "Formatted text of an empty document should be an empty string"

    def test_format_with_no_words(self):
        """Test that the formatter can format a document with no words."""
        formatter = LayoutPreservingFormatter()
        line = Line(words=[])
        document = Document(lines=[line])
        formatted_text = formatter.format(document)

        assert (
            formatted_text == ""
        ), "Formatted text for lines with no words should be an empty string"
