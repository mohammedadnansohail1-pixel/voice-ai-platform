"""Abstract base classes for pipeline components."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class TranscriptionResult:
    """ASR transcription result."""
    text: str
    language: str
    confidence: float
    duration_seconds: float


@dataclass
class SynthesisResult:
    """TTS synthesis result."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float


@dataclass
class VADResult:
    """Voice activity detection result."""
    is_speech: bool
    speech_probability: float
    speech_started: bool = False
    speech_ended: bool = False


class ASRBackend(ABC):
    """Abstract speech-to-text backend."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate
            language: Optional language hint (None = auto-detect)
            
        Returns:
            TranscriptionResult with text, language, confidence
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class TTSBackend(ABC):
    """Abstract text-to-speech backend."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Optional voice ID override
            
        Returns:
            SynthesisResult with audio array and sample rate
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice: str) -> None:
        """Set the voice to use."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class VADBackend(ABC):
    """Abstract voice activity detection backend."""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """
        Process an audio chunk for voice activity.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            VADResult with speech detection info
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new utterance."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class LLMBackend(ABC):
    """Abstract language model backend."""
    
    @abstractmethod
    def load(self) -> None:
        """Initialize connection/load model."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response.
        
        Args:
            prompt: User input
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat with conversation history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if connected/ready."""
        pass
