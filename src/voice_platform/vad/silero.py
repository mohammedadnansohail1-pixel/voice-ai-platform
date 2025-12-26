"""Silero VAD implementation."""
import torch
import numpy as np
from typing import Optional

from ..core.registry import vad_registry
from ..core.config import VADConfig
from ..logging import get_logger
from ..core.exceptions import ModelLoadError, ModelInferenceError
from .base import BaseVAD, VADResult

logger = get_logger("vad.silero")


@vad_registry.register("silero")
class SileroVAD(BaseVAD):
    """
    Silero VAD - ML-based voice activity detection.
    
    4x more accurate than WebRTC VAD, industry standard.
    ~1ms per chunk, ~1MB model size.
    """
    
    def __init__(self, config: Optional[VADConfig] = None) -> None:
        if config is None:
            config = VADConfig()
        super().__init__(config)
        
        self.model = None
        self.sample_rate = 16000
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Silero VAD model."""
        logger.info("loading_silero_vad")
        
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        
        # Get utility functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        
        # Create iterator for streaming
        self.vad_iterator = self.VADIterator(
            self.model,
            threshold=self.config.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.config.min_silence_ms,
            speech_pad_ms=30,
        )
        
        logger.info("silero_vad_loaded", threshold=self.config.threshold)
    
    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> VADResult:
        """
        Process audio chunk through Silero VAD.
        
        Args:
            audio: Float32 audio samples normalized to [-1, 1]
            sample_rate: Must be 16000 for Silero
        
        Returns:
            VADResult with speech probability
        """
        if sample_rate != 16000:
            raise ValueError(f"Silero VAD requires 16kHz audio, got {sample_rate}")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        is_speech = speech_prob >= self.config.threshold
        
        return VADResult(
            is_speech=is_speech,
            confidence=speech_prob,
            audio_data=audio if is_speech else None,
        )
    
    def reset(self) -> None:
        """Reset VAD state for new session."""
        self.vad_iterator.reset_states()
        logger.debug("vad_state_reset")
