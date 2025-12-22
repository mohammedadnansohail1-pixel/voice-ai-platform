"""Voice Activity Detection backends."""
from .base import BaseVAD, VADResult
from .silero import SileroVAD

__all__ = ["BaseVAD", "VADResult", "SileroVAD"]
