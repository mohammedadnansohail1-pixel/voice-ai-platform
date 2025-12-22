"""Platform exceptions."""


class VoicePlatformError(Exception):
    """Base exception for voice platform."""
    pass


class ConfigError(VoicePlatformError):
    """Configuration error."""
    pass


class ChannelError(VoicePlatformError):
    """Channel/transport error."""
    pass


class PipelineError(VoicePlatformError):
    """Audio pipeline error."""
    pass


class IntegrationError(VoicePlatformError):
    """External integration error."""
    pass
