class DetectionFusionError(Exception):
    """Base exception for all DetectionFusion errors."""

    pass


class ConfigurationError(DetectionFusionError):
    """Raised when configuration is invalid or missing."""

    pass


class FormatError(DetectionFusionError):
    """Raised when annotation format is invalid or unsupported."""

    pass
