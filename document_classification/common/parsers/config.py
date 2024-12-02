from pydantic import BaseModel


class ParserConfig(BaseModel):
    """Configuration for the parsers package."""

    merge_threshold: float = 0.53


parser_config = ParserConfig()
