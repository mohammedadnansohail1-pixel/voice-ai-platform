"""Base ASR interface."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..core.config import ASRConfig
from ..core.types import Transcript, TranscriptSegment


class BaseASR(ABC):
    """Abstract base class for ASR backends."""
    
    def __init__(self, config: ASRConfig) -> None:
        self.config = config
    
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Transcribe audio to text.
        
        Args:
            audio: Float32 audio samples
            sample_rate: Sample rate in Hz
            language: Language code (None for auto-detect)
        
        Returns:
            Transcript with text and segments
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> Transcript:
        """
        Transcribe streaming audio chunks.
        
        Args:
            audio_chunks: List of audio chunks
            sample_rate: Sample rate in Hz
        
        Returns:
            Transcript with text and segments
        """
        pass
    
    def bytes_to_float(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes (int16) to float32 numpy array."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0
