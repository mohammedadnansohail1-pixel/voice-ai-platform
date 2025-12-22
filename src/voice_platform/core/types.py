"""
Voice AI Platform - Shared Type Definitions
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = "initializing"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    IDLE = "idle"
    ENDED = "ended"
    ERROR = "error"


class ChannelType(Enum):
    """Supported channel types."""
    LOCAL_MIC = "local_mic"
    WEBSOCKET = "websocket"
    TWILIO = "twilio"
    SIP = "sip"


@dataclass
class AudioChunk:
    """Raw audio data chunk."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    
    def __post_init__(self) -> None:
        if self.duration_ms == 0.0 and self.data:
            # Calculate duration: bytes / (sample_rate * channels * 2 bytes per sample)
            self.duration_ms = (len(self.data) / (self.sample_rate * self.channels * 2)) * 1000


@dataclass
class TranscriptSegment:
    """Speech-to-text result segment."""
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 1.0
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Transcript:
    """Complete transcript with segments."""
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: float = 0.0
    
    @classmethod
    def from_text(cls, text: str, language: Optional[str] = None) -> "Transcript":
        """Create a simple transcript from text."""
        return cls(text=text, language=language)


@dataclass
class LLMMessage:
    """Message for LLM conversation."""
    role: str  # system, user, assistant
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class TTSResult:
    """Text-to-speech result."""
    audio_data: bytes
    sample_rate: int = 24000
    duration_ms: float = 0.0
    voice: Optional[str] = None
    language: Optional[str] = None


@dataclass
class SessionContext:
    """Session state and context."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZING
    channel_type: ChannelType = ChannelType.LOCAL_MIC
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Conversation history
    messages: list[LLMMessage] = field(default_factory=list)
    
    # Current turn tracking
    current_transcript: Optional[Transcript] = None
    is_speaking: bool = False
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append(LLMMessage(role=role, content=content))
    
    def get_conversation_history(self, max_turns: int = 10) -> list[LLMMessage]:
        """Get recent conversation history."""
        return self.messages[-max_turns * 2:] if self.messages else []
    
    @property
    def duration_s(self) -> float:
        """Session duration in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
