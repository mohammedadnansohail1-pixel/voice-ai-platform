"""Abstract channel interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
import numpy as np


class ChannelEventType(str, Enum):
    """Types of channel events."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    AUDIO_RECEIVED = "audio_received"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    ERROR = "error"


@dataclass
class ChannelEvent:
    """Event from a channel."""
    type: ChannelEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 16000
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


# Type alias for event handler
EventHandler = Callable[[ChannelEvent], None]


class Channel(ABC):
    """
    Abstract audio channel.
    
    Handles bidirectional audio communication.
    Implementations: LocalMic, WebSocket, TwilioSIP
    """
    
    def __init__(self, session_id: str, sample_rate: int = 16000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self._event_handlers: list[EventHandler] = []
        self._is_active = False
    
    def on_event(self, handler: EventHandler) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)
    
    def _emit(self, event: ChannelEvent) -> None:
        """Emit an event to all handlers."""
        event.session_id = self.session_id
        for handler in self._event_handlers:
            handler(event)
    
    @property
    def is_active(self) -> bool:
        """Check if channel is active."""
        return self._is_active
    
    @abstractmethod
    def start(self) -> None:
        """Start the channel (begin receiving audio)."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the channel."""
        pass
    
    @abstractmethod
    def play_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """
        Play audio through the channel.
        
        Args:
            audio: Audio samples to play
            sample_rate: Sample rate of the audio
        """
        pass
    
    @abstractmethod
    def interrupt_playback(self) -> None:
        """Interrupt any ongoing audio playback (for barge-in)."""
        pass
