import re


class TextPreprocessor:
    """Handles text preprocessing for uniformity across training and inference."""

    @staticmethod
    def mask_text(text: str) -> str:
        """
        Mask numbers and standardize text format.

        Args:
            text: The input text to be processed.

        Returns:
            Modified text with masked numbers and standardized format.

        """
        text = re.sub(r"\d", "X", text)
        text = re.sub(r"[-@:\/]", " ", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess a single text."""
        return TextPreprocessor.mask_text(text)
