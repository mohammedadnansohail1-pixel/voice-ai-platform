"""Faster-Whisper ASR implementation."""
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from ..core.registry import asr_registry
from ..core.config import ASRConfig
from ..core.types import Transcript, TranscriptSegment
from ..logging import get_logger
from .base import BaseASR

logger = get_logger("asr.whisper")


@asr_registry.register("whisper")
class WhisperASR(BaseASR):
    """
    Faster-Whisper ASR - CTranslate2-optimized Whisper.
    
    4x faster than original Whisper with same accuracy.
    Supports GPU acceleration with float16/int8.
    """
    
    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        if config is None:
            config = ASRConfig()
        super().__init__(config)
        
        self.model: Optional[WhisperModel] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model."""
        logger.info(
            "loading_whisper",
            model=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        
        start = time.perf_counter()
        
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        
        load_time = time.perf_counter() - start
        logger.info("whisper_loaded", load_time_s=f"{load_time:.2f}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Float32 audio samples normalized to [-1, 1]
            sample_rate: Must be 16000 for Whisper
            language: Language code or None for auto-detect
        
        Returns:
            Transcript with segments
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        start = time.perf_counter()
        
        # Use config language if not specified
        lang = language or self.config.language
        
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=30,
            ),
        )
        
        # Collect segments
        transcript_segments = []
        full_text_parts = []
        
        for seg in segments:
            transcript_segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=seg.avg_logprob,
                    language=info.language,
                )
            )
            full_text_parts.append(seg.text.strip())
        
        full_text = " ".join(full_text_parts)
        latency_ms = (time.perf_counter() - start) * 1000
        
        logger.debug(
            "transcription_complete",
            text_length=len(full_text),
            segments=len(transcript_segments),
            language=info.language,
            latency_ms=f"{latency_ms:.1f}",
        )
        
        return Transcript(
            text=full_text,
            segments=transcript_segments,
            language=info.language,
            duration=audio.shape[0] / sample_rate,
        )
    
    def transcribe_stream(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> Transcript:
        """
        Transcribe accumulated audio chunks.
        
        Args:
            audio_chunks: List of float32 audio chunks
            sample_rate: Sample rate in Hz
        
        Returns:
            Transcript
        """
        # Concatenate chunks
        if not audio_chunks:
            return Transcript(text="", segments=[], duration=0.0)
        
        audio = np.concatenate(audio_chunks)
        return self.transcribe(audio, sample_rate)
