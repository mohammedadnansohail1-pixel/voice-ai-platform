"""Kokoro TTS backend."""
from typing import Optional
import numpy as np

from .base import TTSBackend, SynthesisResult
from .registries import tts_registry
from ..logging import get_logger

logger = get_logger("tts.kokoro")

# Kokoro language codes (single character)
LANG_MAP = {
    "a": "a",   # American English
    "b": "b",   # British English
    "e": "e",   # Spanish
    "f": "f",   # French
    "h": "h",   # Hindi
    "i": "i",   # Italian
    "p": "p",   # Portuguese
    "j": "j",   # Japanese
    "z": "z",   # Mandarin Chinese
}


@tts_registry.register("kokoro")
class KokoroTTS(TTSBackend):
    """Kokoro TTS backend."""
    
    def __init__(
        self,
        voice: str = "af_heart",
        device: str = "cuda",
        sample_rate: int = 24000,
    ):
        self._voice = voice
        self.device = device
        self.sample_rate = sample_rate
        self._pipeline = None
        self._current_lang = None
    
    def _get_lang_code(self, voice: str) -> str:
        """Extract single-char lang code from voice name."""
        # Voice format: af_heart, bf_emma, etc.
        # First char is the language code
        if voice and len(voice) >= 1:
            return voice[0]
        return "a"  # Default to American English
    
    def load(self) -> None:
        """Load the Kokoro model."""
        if self._pipeline is not None:
            return
        
        from kokoro import KPipeline
        
        lang_code = self._get_lang_code(self._voice)
        
        logger.info("loading_kokoro", voice=self._voice, lang=lang_code, device=self.device)
        
        self._pipeline = KPipeline(lang_code=lang_code, device=self.device)
        self._current_lang = lang_code
        logger.info("kokoro_loaded")
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> SynthesisResult:
        """Synthesize speech from text."""
        if not self.is_loaded:
            self.load()
        
        use_voice = voice or self._voice
        
        # Check if we need to reload for different language
        new_lang = self._get_lang_code(use_voice)
        if new_lang != self._current_lang:
            from kokoro import KPipeline
            logger.info("switching_language", from_lang=self._current_lang, to_lang=new_lang)
            self._pipeline = KPipeline(lang_code=new_lang, device=self.device)
            self._current_lang = new_lang
        
        logger.debug("synthesizing", text=text[:50], voice=use_voice)
        
        # Generate audio
        audio_segments = []
        for result in self._pipeline(text, voice=use_voice):
            if result.audio is not None:
                audio_segments.append(result.audio.numpy())
        
        if not audio_segments:
            # Return silence if nothing generated
            audio = np.zeros(self.sample_rate, dtype=np.float32)
        else:
            audio = np.concatenate(audio_segments)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        duration = len(audio) / self.sample_rate
        
        logger.debug("synthesized", duration=f"{duration:.2f}s", samples=len(audio))
        
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
        )
    
    def set_voice(self, voice: str) -> None:
        """Set the voice to use."""
        self._voice = voice
    
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
