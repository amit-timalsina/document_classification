from __future__ import annotations


class BaseError(Exception):
    """Base exception for all Kniru-related errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred. Please contact support."

    def __init__(self, message: str | None = None, code: int | None = None) -> None:
        """
        Initialize the KniruError.

        Args:
        ----
            message (Optional[str]): The error message. If None, uses the default detail.
            code (Optional[int]): The HTTP status code. If None, uses the default status_code.

        """
        self.message = message or self.detail
        self.code = code or self.status_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Return a string representation of the error.

        Returns
        -------
            str: A formatted error message.

        """
        return f"BaseError: {self.message} (Code: {self.code})"
