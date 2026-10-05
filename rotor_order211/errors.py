"""Domain errors carried to the HTTP layer as 422 responses."""


class AnalysisError(ValueError):
    """Raised when input data or parameters violate analysis constraints."""
