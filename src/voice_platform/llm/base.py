"""Base LLM interface."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ..core.config import LLMConfig
from ..core.types import LLMMessage, LLMResponse


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
    
    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response synchronously.
        
        Args:
            messages: Conversation history
            max_tokens: Override default max tokens
        
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a response with streaming.
        
        Args:
            messages: Conversation history
            max_tokens: Override default max tokens
        
        Yields:
            Token strings as they're generated
        """
        pass
    
    def _build_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Convert LLMMessage objects to dict format."""
        result = []
        
        # Add system prompt if configured
        if self.config.system_prompt:
            result.append({
                "role": "system",
                "content": self.config.system_prompt,
            })
        
        # Add conversation messages
        for msg in messages:
            result.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return result
