from __future__ import annotations

from typing import TYPE_CHECKING

from document_classification.common.parsers.pydantic_models import Document, Line, Word

if TYPE_CHECKING:
    import pandas as pd


class DefaultParser:
    """A parser that converts DataFrame into Document by grouping words into lines."""

    def __init__(self, merge_threshold: float = 0.53) -> None:
        """
        Initialize DefaultParser with merge threshold for line detection.

        Args:
            merge_threshold (float, optional): Threshold to determine if words belong to same line.
            Defaults to 0.53.

        """
        self.merge_threshold = merge_threshold

    def parse(self, df: pd.DataFrame) -> Document:
        """
        Parse DataFrame into Document by grouping words into lines.

        Args:
            df (pd.DataFrame): DataFrame containing text and coordinates of words.

        Returns:
            Document: Parsed document containing lines of words.

        """
        words = self._df_to_words(df)
        lines = self._words_to_lines(words)
        return Document(lines=lines)

    def _df_to_words(self, df: pd.DataFrame) -> list[Word]:
        return [
            Word(text=row["text"], x0=row["x0"], y0=row["y0"], x2=row["x2"], y2=row["y2"])
            for _, row in df.iterrows()
        ]

    def _words_to_lines(self, words: list[Word]) -> list[Line]:
        sorted_words = sorted(words, key=lambda w: (w.y0, w.x0))
        lines = []
        current_line: list[Word] = []
        last_y0 = None

        for word in sorted_words:
            if self._is_new_line(word, last_y0):
                if current_line:
                    lines.append(Line(words=current_line))
                current_line = [word]
                last_y0 = word.y0
            else:
                current_line.append(word)

        if current_line:
            lines.append(Line(words=current_line))

        return lines

    def _is_new_line(self, word: Word, last_y0: float | None) -> bool:
        return last_y0 is None or (word.y0 - last_y0) > self.merge_threshold * (word.y2 - word.y0)
