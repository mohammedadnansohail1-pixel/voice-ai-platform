"""Text-to-speech backends."""
from .base import BaseTTS, TTSResult
from .kokoro import KokoroTTS
from .piper import PiperTTS

__all__ = ["BaseTTS", "TTSResult", "KokoroTTS", "PiperTTS"]
