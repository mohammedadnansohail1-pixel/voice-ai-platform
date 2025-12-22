"""Whisper ASR backend using faster-whisper."""
from typing import Optional
import numpy as np

from .base import ASRBackend, TranscriptionResult
from .registries import asr_registry
from ..logging import get_logger

logger = get_logger("asr.whisper")


@asr_registry.register("whisper")
class WhisperASR(ASRBackend):
    """Faster-Whisper ASR backend."""
    
    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",  # Force English
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
    
    def load(self) -> None:
        if self._model is not None:
            return
        
        from faster_whisper import WhisperModel
        
        logger.info("loading_whisper", model=self.model_name, device=self.device)
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("whisper_loaded")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if not self.is_loaded:
            self.load()
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Use provided language or default
        use_language = language or self.language
        
        segments, info = self._model.transcribe(
            audio,
            language=use_language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = " ".join(text_parts)
        duration = len(audio) / sample_rate
        
        logger.debug("transcribed", text=text[:50], language=info.language)
        
        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=info.language_probability,
            duration_seconds=duration,
        )
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
