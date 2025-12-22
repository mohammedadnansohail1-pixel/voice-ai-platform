"""Piper TTS implementation - fast multilingual TTS."""
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from ..core.registry import tts_registry
from ..core.config import TTSConfig
from ..core.exceptions import ModelLoadError
from ..logging import get_logger
from .base import BaseTTS, TTSResult

logger = get_logger("tts.piper")


@tts_registry.register("piper")
class PiperTTS(BaseTTS):
    """
    Piper TTS - Fast, lightweight multilingual TTS.
    
    Good for Arabic, Hindi, and other languages.
    Runs on CPU efficiently.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        if config is None:
            config = TTSConfig(backend="piper")
        super().__init__(config)
        
        self.sample_rate = 22050
        self.models_dir = Path.home() / ".local" / "share" / "piper" / "models"
        self._verify_installation()
    
    def _verify_installation(self) -> None:
        """Verify Piper is installed."""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ModelLoadError("tts", "piper", "piper not found in PATH")
            logger.info("piper_verified", version=result.stdout.strip())
        except FileNotFoundError:
            raise ModelLoadError("tts", "piper", "piper not installed")
    
    def _get_model_path(self, voice: str) -> Path:
        """Get path to voice model."""
        # Voice format: lang_REGION-name-quality (e.g., ar_JO-kareem-medium)
        model_file = self.models_dir / f"{voice}.onnx"
        if not model_file.exists():
            # Try downloading or raise error
            logger.warning("piper_model_not_found", voice=voice, path=str(model_file))
        return model_file
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech using Piper."""
        start = time.perf_counter()
        
        # Select voice based on language if provided
        if language and not voice:
            voice = self.get_voice_for_language(language)
        voice = voice or self.config.voice
        
        model_path = self._get_model_path(voice)
        
        try:
            # Run Piper as subprocess
            process = subprocess.run(
                [
                    "piper",
                    "--model", str(model_path),
                    "--output_raw",
                ],
                input=text,
                capture_output=True,
                text=True,
                check=True,
            )
            
            # Parse raw audio output (int16)
            audio_bytes = process.stdout.encode("latin-1")
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
            
        except subprocess.CalledProcessError as e:
            logger.error("piper_failed", error=e.stderr)
            # Return silence on error
            audio = np.zeros(self.sample_rate // 10, dtype=np.float32)
        except Exception as e:
            logger.error("piper_error", error=str(e))
            audio = np.zeros(self.sample_rate // 10, dtype=np.float32)
        
        duration_ms = (len(audio) / self.sample_rate) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        
        logger.debug(
            "piper_synthesized",
            text_length=len(text),
            duration_ms=f"{duration_ms:.1f}",
            latency_ms=f"{latency_ms:.1f}",
        )
        
        return TTSResult(
            audio_data=audio,
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
            voice=voice,
            language=language,
        )
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncIterator[np.ndarray]:
        """Stream synthesis - Piper doesn't support streaming, so yield full result."""
        result = self.synthesize(text, voice)
        yield result.audio_data
