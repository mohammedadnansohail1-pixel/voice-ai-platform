"""Local microphone channel using sounddevice."""
import threading
import queue
import time
from typing import Optional, Callable
import numpy as np

from .base import Channel, ChannelEvent, ChannelEventType
from .registry import channel_registry
from ..logging import get_logger

logger = get_logger("channel.local_mic")


@channel_registry.register("local_mic")
class LocalMicChannel(Channel):
    """
    Local microphone and speaker channel.
    
    Uses sounddevice for audio I/O.
    Emits AUDIO_RECEIVED events with audio chunks.
    """
    
    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        super().__init__(session_id, sample_rate)
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.input_device = input_device
        self.output_device = output_device
        
        self._input_stream = None
        self._output_stream = None
        self._playback_queue: queue.Queue = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()
        self._is_playing = threading.Event()
    
    def start(self) -> None:
        """Start capturing audio from microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
        
        if self._is_active:
            return
        
        logger.info("starting_local_mic", session_id=self.session_id)
        
        # Start input stream
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            device=self.input_device,
            callback=self._audio_callback,
        )
        self._input_stream.start()
        
        # Start playback thread
        self._stop_playback.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        
        self._is_active = True
        self._emit(ChannelEvent(type=ChannelEventType.CONNECTED))
        logger.info("local_mic_started", session_id=self.session_id)
    
    def stop(self) -> None:
        """Stop the channel."""
        if not self._is_active:
            return
        
        logger.info("stopping_local_mic", session_id=self.session_id)
        
        self._is_active = False
        
        # Stop input stream
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        
        # Stop playback thread
        self._stop_playback.set()
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
        
        self._emit(ChannelEvent(type=ChannelEventType.DISCONNECTED))
        logger.info("local_mic_stopped", session_id=self.session_id)
    
    def play_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Queue audio for playback."""
        if not self._is_active:
            return
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        self._playback_queue.put(audio)
    
    def interrupt_playback(self) -> None:
        """Stop current playback (for barge-in)."""
        # Clear the queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        
        # Signal to stop current playback
        self._stop_playback.set()
        time.sleep(0.05)  # Brief pause
        self._stop_playback.clear()
        
        logger.debug("playback_interrupted", session_id=self.session_id)
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing.is_set()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice when audio is received."""
        if status:
            logger.warning("audio_callback_status", status=str(status))
        
        # Emit audio event
        audio = indata[:, 0].copy()  # Mono
        self._emit(ChannelEvent(
            type=ChannelEventType.AUDIO_RECEIVED,
            audio=audio,
            sample_rate=self.sample_rate,
        ))
    
    def _playback_loop(self):
        """Background thread for audio playback."""
        try:
            import sounddevice as sd
        except ImportError:
            return
        
        while not self._stop_playback.is_set() or not self._playback_queue.empty():
            try:
                audio = self._playback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if self._stop_playback.is_set():
                continue
            
            self._is_playing.set()
            try:
                sd.play(audio, self.sample_rate, device=self.output_device)
                sd.wait()
            except Exception as e:
                logger.error("playback_error", error=str(e))
            finally:
                self._is_playing.clear()
    
    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if from_sr == to_sr:
            return audio
        
        duration = len(audio) / from_sr
        new_length = int(duration * to_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
