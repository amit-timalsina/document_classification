import pandas as pd

from common.parsers.interfaces import Formatter, Parser


def parse_and_format(df: pd.DataFrame, parser: Parser, formatter: Formatter) -> str:
    """
    Parse and format a DataFrame using the provided parser and formatter.

    Args:
        df (pd.DataFrame): The input DataFrame to be parsed and formatted.
        parser (Parser): The parser object used to parse the DataFrame.
        formatter (Formatter): The formatter object used to format the parsed document.

    Returns:
        str: The formatted document as a string.

    """
    document = parser.parse(df)
    return formatter.format(document)
