"""Base channel interface."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import numpy as np

from ..core.config import Config
from ..core.types import SessionContext


class BaseChannel(ABC):
    """Abstract base for audio channels (WebSocket, Twilio, etc.)."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session: Optional[SessionContext] = None
    
    @abstractmethod
    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """Receive audio chunks from client."""
        pass
    
    @abstractmethod
    async def send_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Send audio to client."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the channel."""
        pass
