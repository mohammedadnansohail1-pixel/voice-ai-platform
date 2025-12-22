"""Base TTS interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

from ..core.config import TTSConfig


@dataclass
class TTSResult:
    """TTS synthesis result."""
    audio_data: np.ndarray  # float32 audio samples
    sample_rate: int
    duration_ms: float
    voice: str
    language: Optional[str] = None


class BaseTTS(ABC):
    """Abstract base class for TTS backends."""
    
    def __init__(self, config: TTSConfig) -> None:
        self.config = config
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (None for default)
            language: Language code for voice selection
        
        Returns:
            TTSResult with audio data
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncIterator[np.ndarray]:
        """
        Stream synthesized audio chunks.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (None for default)
        
        Yields:
            Audio chunks as numpy arrays
        """
        pass
    
    def get_voice_for_language(self, language: str) -> str:
        """Get configured voice for a language."""
        return self.config.voices.get(language, self.config.voice)
    
    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to int16 bytes."""
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
