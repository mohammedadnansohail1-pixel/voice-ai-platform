"""Kokoro TTS implementation - high quality English voices."""
import time
from typing import AsyncIterator, Optional

import numpy as np

from ..core.registry import tts_registry
from ..core.config import TTSConfig
from ..core.exceptions import ModelLoadError
from ..logging import get_logger
from .base import BaseTTS, TTSResult

logger = get_logger("tts.kokoro")


@tts_registry.register("kokoro")
class KokoroTTS(BaseTTS):
    """
    Kokoro TTS - High quality neural TTS.
    
    Best for English, fast inference on GPU.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        if config is None:
            config = TTSConfig()
        super().__init__(config)
        
        self.pipeline = None
        self.sample_rate = 24000
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Kokoro TTS model."""
        logger.info("loading_kokoro", voice=self.config.voice, device=self.config.device)
        
        try:
            from kokoro import KPipeline
            
            start = time.perf_counter()
            
            # Determine language code from voice
            lang_code = "a"  # American English default
            if self.config.voice.startswith("b"):
                lang_code = "b"  # British English
            
            self.pipeline = KPipeline(lang_code=lang_code)
            self.device = self.config.device
            
            load_time = time.perf_counter() - start
            logger.info("kokoro_loaded", load_time_s=f"{load_time:.2f}")
            
        except ImportError as e:
            raise ModelLoadError("tts", "kokoro", f"kokoro not installed: {e}")
        except Exception as e:
            raise ModelLoadError("tts", "kokoro", str(e))
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech using Kokoro."""
        start = time.perf_counter()
        
        voice = voice or self.config.voice
        
        # Generate audio
        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=voice):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            # Return silence if no audio generated
            audio = np.zeros(self.sample_rate // 10, dtype=np.float32)
        else:
            audio = np.concatenate(audio_chunks)
        
        duration_ms = (len(audio) / self.sample_rate) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        logger.debug(
            "kokoro_synthesized",
            text_length=len(text),
            duration_ms=f"{duration_ms:.1f}",
            latency_ms=f"{latency_ms:.1f}",
        )
        
        return TTSResult(
            audio_data=audio,
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
            voice=voice,
            language=language or "en",
        )
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncIterator[np.ndarray]:
        """Stream synthesized audio chunks."""
        voice = voice or self.config.voice
        
        for _, _, audio in self.pipeline(text, voice=voice):
            yield audio
