from __future__ import annotations

from typing import Literal

import pytest

from document_classification.common.schemas.word import Word


class TestWord:
    """Test suite for the Word class."""

    @pytest.mark.parametrize(
        ("x0", "y0", "x2", "y2"),
        [
            (0, 0, 10, 10),
            (10, 10, 20, 20),
            (20, 20, 30, 30),
        ],
    )
    def test_word_init(
        self,
        x0: Literal[0, 10, 20],
        y0: Literal[0, 10, 20],
        x2: Literal[10, 20, 30],
        y2: Literal[10, 20, 30],
    ):
        """Test that Word is initialized correctly."""
        word = Word(text="test", x0=x0, y0=y0, x2=x2, y2=y2)
        assert word.text == "test"
        assert word.x0 == x0
        assert word.y0 == y0
        assert word.x2 == x2
        assert word.y2 == y2

    @pytest.mark.parametrize(
        ("x0", "y0", "x2", "y2"),
        [
            (-1, 0, 10, 10),
            (0, -1, 10, 10),
            (10, 10, -1, 10),
            (10, 10, 10, -1),
        ],
    )
    def test_word_init_validation(
        self,
        x0: Literal[-1, 0, 10],
        y0: Literal[0, -1, 10],
        x2: Literal[10, -1],
        y2: Literal[10, -1],
    ):
        """Test that Word validation raises a ValueError when coordinates are invalid."""
        with pytest.raises(ValueError):  # noqa: PT011
            Word(text="test", x0=x0, y0=y0, x2=x2, y2=y2)
