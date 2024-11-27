from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from document_classification.common.schemas.line import Line

if TYPE_CHECKING:
    from document_classification.common.schemas.word import Word


@pytest.fixture
def lines(words: list[Word]):
    """Fixture of lines for Document schema tests."""
    return [Line(words=words)]
