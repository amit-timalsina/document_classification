from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def create_enum_for_document_types(
    directory: Path | None = None,
    enum_names: list[str] | None = None,
) -> type[Enum]:
    """
    Dynamically create a DocumentType enum based on subdirectory names or a given list of strings.

    Args:
        directory: The directory containing subdirectories for document types.
        enum_names: A list of strings to use as enum names instead of using subdirectories.

    Returns:
        Type[Enum]: A dynamically created Enum class for document types.

    """
    enum_dict = {}
    if directory is not None and enum_names is None:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                enum_dict[subdir.name] = subdir.name
    elif enum_names is not None:
        enum_dict = {name: name for name in enum_names}
    return Enum("DocumentType", enum_dict)  # type: ignore[return-value]
