from document_classification.common.parsers.config import parser_config
from document_classification.common.schemas import Document, Line


class LayoutPreservingFormatter:
    """
    A formatter that preserves the layout of text based on pixel coordinates.

    This class formats document into a string while maintaining the original spatial layout
    by converting pixel coordinates to character positions.
    """

    def __init__(
        self,
        max_line_length: int = parser_config.max_line_length_char,
        pixel_to_char_ratio: float = parser_config.pixel_to_char_ratio,
    ) -> None:
        """
        Initialize the formatter with max_line_length and pixel to character ratio.

        Args:
            max_line_length: The maximum line length in characters.
            pixel_to_char_ratio: The conversion ratio from pixels to characters.

        """
        self.max_line_length = max_line_length
        self.pixel_to_char_ratio = pixel_to_char_ratio

    def format(self, document: Document) -> str:
        """
        Format the document while preserving the original layout.

        Args:
            document: The Document object containing lines to be formatted.

        Returns:
            A formatted string with preserved spatial layout.

        """
        return "\n".join(self._format_line(line) for line in document.lines)

    def _format_line(self, line: Line) -> str:
        final_string = [" "] * self.max_line_length
        for word in line.words:
            start_index = round(word.x0 * self.pixel_to_char_ratio)
            word_text = word.text
            len_word = len(word_text)
            final_string[start_index : start_index + len_word] = word_text
        return "".join(final_string).rstrip()
