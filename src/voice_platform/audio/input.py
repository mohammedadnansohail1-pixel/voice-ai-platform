"""Audio input capture."""
import threading
import queue
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from ..core.config import AudioConfig
from ..core.types import AudioChunk
from ..logging import get_logger

logger = get_logger("audio.input")


class AudioInput:
    """
    Microphone audio capture with callback support.
    
    Captures audio in chunks and passes to callback for processing.
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.on_audio = on_audio
        
        self.sample_rate = self.config.sample_rate
        self.channels = self.config.channels
        self.chunk_samples = int(self.sample_rate * self.config.chunk_duration_ms / 1000)
        
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        
        logger.info(
            "audio_input_initialized",
            sample_rate=self.sample_rate,
            chunk_ms=self.config.chunk_duration_ms,
            chunk_samples=self.chunk_samples,
        )
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning("audio_input_status", status=str(status))
        
        # Convert to float32 and flatten
        audio = indata[:, 0].astype(np.float32)
        
        # Put in queue for processing
        self.audio_queue.put(audio.copy())
        
        # Call callback if provided
        if self.on_audio:
            self.on_audio(audio)
    
    def start(self) -> None:
        """Start audio capture."""
        if self.is_running:
            return
        
        logger.info("audio_input_starting")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        
        self.stream.start()
        self.is_running = True
        logger.info("audio_input_started")
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self.is_running:
            return
        
        logger.info("audio_input_stopping")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_running = False
        logger.info("audio_input_stopped")
    
    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self) -> None:
        """Clear pending audio chunks."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def __enter__(self) -> "AudioInput":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
