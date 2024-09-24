class CustomException(Exception):
    """
    Custom exception class that logs the error message.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"CustomException: {self.message}"
