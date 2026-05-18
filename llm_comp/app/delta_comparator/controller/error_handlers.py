"""This file is for error handling."""


class InvalidFileException(Exception):
    """InvalidException class."""

    def __init__(self, message: str = "Invalid file type"):
        """Init for InvalidException."""
        super().__init__(message)


class FileRemovalException(Exception):
    """FileRemovalException class."""

    def __init__(self, message: str = "Error removing file"):
        """Init for FileRemovalException."""
        super().__init__(message)


class UnexpectedErrorException(Exception):
    """UnexpectedErrorException class."""

    def __init__(self, message: str = "Core module error"):
        """Init for UnexpectedErrorException."""
        super().__init__(message)


class InvalidInputException(Exception):
    """InvalidInputException class."""

    def __init__(self, message: str = "An unexpected error occurred"):
        """Init InvalidInputException."""
        super().__init__(message)


class InputFormatException(Exception):
    """InputFormatException class."""
