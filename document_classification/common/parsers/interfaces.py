from typing import Protocol

import pandas as pd

from document_classification.common.parsers.pydantic_models import Document


class Parser(Protocol):
    """
    Protocol defining the interface for document parsers.

    A Parser is responsible for converting a pandas DataFrame containing
    document information into a structured Document object.

    Implementations of this protocol should define the parsing logic
    to extract words and lines from the input DataFrame.
    """

    def parse(self, df: pd.DataFrame) -> Document:
        """
        Parse the input DataFrame and return a Document object.

        Args:
            df (pd.DataFrame): A DataFrame containing document information.
                Expected columns include 'text', 'x0', 'y0', 'x2', and 'y2'.

        Returns:
            Document: A structured representation of the parsed document,
                containing a list of Line objects, each with a list of Word objects.

        Raises:
            ValueError: If the input DataFrame doesn't have the required columns
                or if the data is in an unexpected format.

        """
        ...


class Formatter(Protocol):
    """
    Protocol defining the interface for document formatters.

    A Formatter is responsible for converting a structured Document object
    into a formatted string representation, typically preserving the layout
    of the original document.

    Implementations of this protocol should define the formatting logic
    to convert the Document structure into the desired string output.
    """

    def format(self, document: Document) -> str:
        """
        Format the input Document object into a string representation.

        Args:
            document (Document): A structured representation of a document,
                containing a list of Line objects, each with a list of Word objects.

        Returns:
            str: A formatted string representation of the document,
                typically preserving the original layout.

        Raises:
            ValueError: If the input Document object is malformed or
                contains unexpected data.

        """
        ...
