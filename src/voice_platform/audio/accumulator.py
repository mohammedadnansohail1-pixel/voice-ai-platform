"""Speech accumulator - collects speech segments between silences."""
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.config import VADConfig
from ..vad.base import VADResult
from ..logging import get_logger

logger = get_logger("audio.accumulator")


@dataclass
class SpeechSegment:
    """Accumulated speech segment."""
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int = 16000
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class SpeechAccumulator:
    """
    Accumulates speech audio between silence periods.
    
    Uses VAD results to detect speech boundaries and collects
    audio until silence threshold is reached.
    """
    
    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig()
        
        # Timing thresholds
        self.min_speech_ms = self.config.min_speech_ms
        self.min_silence_ms = self.config.min_silence_ms
        self.max_speech_s = self.config.max_speech_s
        
        # State
        self.chunks: list[np.ndarray] = []
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.is_speaking = False
        
        logger.info(
            "accumulator_initialized",
            min_speech_ms=self.min_speech_ms,
            min_silence_ms=self.min_silence_ms,
        )
    
    def process(self, audio: np.ndarray, vad_result: VADResult) -> Optional[SpeechSegment]:
        """
        Process audio chunk with VAD result.
        
        Args:
            audio: Audio chunk
            vad_result: VAD detection result
        
        Returns:
            SpeechSegment if speech ended, None otherwise
        """
        now = time.time()
        
        if vad_result.is_speech:
            # Speech detected
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = now
                self.chunks = []
                logger.debug("speech_started")
            
            self.chunks.append(audio)
            self.last_speech_time = now
            
            # Check max duration
            if self.speech_start_time:
                duration = now - self.speech_start_time
                if duration >= self.max_speech_s:
                    logger.debug("speech_max_duration", duration_s=duration)
                    return self._finalize_segment()
        
        else:
            # Silence detected
            if self.is_speaking and self.last_speech_time:
                silence_duration_ms = (now - self.last_speech_time) * 1000
                
                # Still collect audio during short silence (padding)
                if silence_duration_ms < self.min_silence_ms:
                    self.chunks.append(audio)
                else:
                    # Silence long enough - finalize segment
                    return self._finalize_segment()
        
        return None
    
    def _finalize_segment(self) -> Optional[SpeechSegment]:
        """Finalize and return accumulated speech segment."""
        if not self.chunks or not self.speech_start_time:
            self.reset()
            return None
        
        # Check minimum duration
        duration_ms = (time.time() - self.speech_start_time) * 1000
        if duration_ms < self.min_speech_ms:
            logger.debug("speech_too_short", duration_ms=duration_ms)
            self.reset()
            return None
        
        # Concatenate chunks
        audio = np.concatenate(self.chunks)
        
        segment = SpeechSegment(
            audio=audio,
            start_time=self.speech_start_time,
            end_time=time.time(),
        )
        
        logger.debug(
            "speech_segment_complete",
            duration_ms=f"{segment.duration_ms:.0f}",
            samples=len(audio),
        )
        
        self.reset()
        return segment
    
    def reset(self) -> None:
        """Reset accumulator state."""
        self.chunks = []
        self.speech_start_time = None
        self.last_speech_time = None
        self.is_speaking = False
    
    def force_finalize(self) -> Optional[SpeechSegment]:
        """Force finalize current segment (e.g., on timeout)."""
        if self.is_speaking and self.chunks:
            return self._finalize_segment()
        return None
