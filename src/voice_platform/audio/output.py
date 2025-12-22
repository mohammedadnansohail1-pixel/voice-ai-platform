"""Audio output playback."""
import threading
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from ..logging import get_logger

logger = get_logger("audio.output")


class AudioOutput:
    """
    Speaker audio playback with queue support.
    
    Supports interruption for barge-in.
    """
    
    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.stream: Optional[sd.OutputStream] = None
        self.is_playing = False
        self.should_stop = False
        
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.play_thread: Optional[threading.Thread] = None
        
        logger.info("audio_output_initialized", sample_rate=sample_rate)
    
    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio synchronously (blocking).
        
        Args:
            audio: Float32 audio samples
            sample_rate: Sample rate (uses default if None)
        """
        sr = sample_rate or self.sample_rate
        
        if self.should_stop:
            return
        
        self.is_playing = True
        try:
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            logger.error("audio_play_error", error=str(e))
        finally:
            self.is_playing = False
    
    def play_async(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio asynchronously (non-blocking).
        
        Args:
            audio: Float32 audio samples
            sample_rate: Sample rate (uses default if None)
        """
        sr = sample_rate or self.sample_rate
        
        def _play():
            self.is_playing = True
            try:
                sd.play(audio, sr)
                sd.wait()
            except Exception as e:
                logger.error("audio_play_error", error=str(e))
            finally:
                self.is_playing = False
        
        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()
    
    def stop(self) -> None:
        """Stop current playback immediately (for barge-in)."""
        self.should_stop = True
        sd.stop()
        self.is_playing = False
        logger.debug("audio_playback_stopped")
    
    def reset(self) -> None:
        """Reset for new playback."""
        self.should_stop = False
    
    def wait(self) -> None:
        """Wait for current playback to finish."""
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join()
