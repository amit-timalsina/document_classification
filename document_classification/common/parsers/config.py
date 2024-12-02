from pydantic import BaseModel


class ParserConfig(BaseModel):
    """Configuration for the parsers package."""

    merge_threshold: float = 0.53
    max_line_length_char: int = 300
    pixel_to_char_ratio: float = 0.2


parser_config = ParserConfig()
