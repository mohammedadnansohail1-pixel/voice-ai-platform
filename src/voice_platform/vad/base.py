"""Base VAD interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.config import VADConfig


@dataclass
class VADResult:
    """Voice activity detection result."""
    is_speech: bool
    confidence: float  # 0.0 - 1.0
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    
    # For accumulating speech segments
    audio_data: Optional[np.ndarray] = field(default=None, repr=False)


class BaseVAD(ABC):
    """Abstract base class for VAD backends."""
    
    def __init__(self, config: VADConfig) -> None:
        self.config = config
    
    @abstractmethod
    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> VADResult:
        """
        Process a single audio chunk.
        
        Args:
            audio: Audio samples as float32 numpy array
            sample_rate: Sample rate in Hz
        
        Returns:
            VADResult with speech detection info
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new session."""
        pass
    
    def bytes_to_float(self, audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Convert raw audio bytes (int16) to float32 numpy array."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0
