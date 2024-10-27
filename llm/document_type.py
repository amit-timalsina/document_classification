from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def create_document_type_enum(directory: Path, subset: list[str] | None = None) -> type[Enum]:
    """
    Dynamically create a DocumentType enum based on subdirectory names.

    Args:
        directory: The directory containing subdirectories for document types.
        subset: A subset of subdirectory names to include. If None, include all.

    Returns:
        Type[Enum]: A dynamically created Enum class for document types.

    """
    enum_dict = {}
    for subdir in directory.iterdir():
        if subdir.is_dir() and (subset is None or subdir.name in subset):
            enum_dict[subdir.name] = subdir.name
    return Enum("DocumentType", enum_dict)  # type: ignore[return-value]
