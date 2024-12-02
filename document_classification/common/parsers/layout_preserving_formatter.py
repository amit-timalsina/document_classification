from document_classification.common.schemas import Document, Line


class LayoutPreservingFormatter:
    """
    A formatter that preserves the layout of text based on pixel coordinates.

    This class formats document into a string while maintaining the original spatial layout
    by converting pixel coordinates to character positions.
    """

    def __init__(self, size: int = 300, pixel_to_char: float = 0.2) -> None:
        """
        Initialize the formatter with size and pixel to character ratio.

        Args:
            size: The maximum line length in characters.
            pixel_to_char: The conversion ratio from pixels to characters.

        """
        self.size = size
        self.pixel_to_char = pixel_to_char

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
        final_string = [" "] * self.size
        for word in line.words:
            start_index = round(word.x0 * self.pixel_to_char)
            word_text = word.text
            len_word = len(word_text)
            final_string[start_index : start_index + len_word] = word_text
        return "".join(final_string).rstrip()
