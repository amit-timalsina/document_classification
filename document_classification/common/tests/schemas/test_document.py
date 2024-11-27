from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.common.schemas import Document

if TYPE_CHECKING:
    from document_classification.common.schemas.line import Line


class TestDocument:
    """Test suite for Document schema."""

    def test_document_init(self, lines: list[Line]):
        """Test that Document is initialized correctly."""
        document = Document(lines=lines)
        assert document.lines == lines

    def test_document_empty_init(self):
        """Test that Document can be initialized with an empty list of lines."""
        document = Document()
        assert document.lines == []
