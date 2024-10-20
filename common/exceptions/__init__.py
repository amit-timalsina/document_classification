"""
Common exception classes for the application.

This package contains custom exception classes that are used throughout the application
to handle specific error scenarios. These exceptions help in providing more detailed
and context-specific error handling, improving the overall robustness and maintainability
of the codebase.

Exception Hierarchy:
All custom exceptions in this application should inherit from the base `BaseError` class.
This creates a consistent exception hierarchy and allows for easier error handling and
identification of application-specific exceptions.

Usage:
To use these exceptions in your code, import them from this package and raise them
when appropriate.

Creating New Exceptions:
When creating new exception classes, follow these guidelines:
1. Inherit from BaseError or an appropriate subclass.
2. Use descriptive names that end with 'Error'.
3. Provide a clear and informative docstring.
4. If needed, implement custom attributes or methods.

Example:
-------
    class CustomError(BaseError):
        '''A custom error that occurs in specific situations.'''
        def __init__(self, message: str, additional_info: str = None):
            super().__init__(message)
            self.additional_info = additional_info

"""

from .base_error import BaseError

__all__ = ("BaseError",)
