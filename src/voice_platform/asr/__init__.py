"""Speech-to-text backends."""
from .base import BaseASR
from .whisper import WhisperASR

__all__ = ["BaseASR", "WhisperASR"]
