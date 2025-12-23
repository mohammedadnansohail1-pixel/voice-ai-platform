"""Ollama LLM implementation with streaming."""
import asyncio
import json
import time
from typing import AsyncIterator, Callable, Optional

import httpx

from ..core.registry import llm_registry
from ..core.config import LLMConfig
from ..core.types import LLMMessage, LLMResponse
from ..core.exceptions import ModelInferenceError
from ..logging import get_logger
from .base import BaseLLM

logger = get_logger("llm.ollama")


@llm_registry.register("ollama")
class OllamaLLM(BaseLLM):
    """
    Ollama LLM backend with streaming support.
    
    Streams tokens and buffers by sentence for TTS handoff.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        if config is None:
            config = LLMConfig()
        super().__init__(config)
        
        self.base_url = config.base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
        self.async_client: Optional[httpx.AsyncClient] = None
        
        logger.info(
            "ollama_initialized",
            model=self.config.model,
            base_url=self.base_url,
        )
    
    def _ensure_async_client(self) -> httpx.AsyncClient:
        """Lazy init async client."""
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(timeout=60.0)
        return self.async_client
    
    def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response synchronously."""
        start = time.perf_counter()
        
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(messages),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
        except httpx.HTTPError as e:
            logger.error("ollama_request_failed", error=str(e))
            raise ModelInferenceError(
                model=self.config.model,
                operation="generate",
                reason=str(e),
            )
        
        latency_ms = (time.perf_counter() - start) * 1000
        content = data.get("message", {}).get("content", "")
        
        logger.debug(
            "llm_response",
            content_length=len(content),
            latency_ms=f"{latency_ms:.1f}",
        )
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            tokens_used=data.get("eval_count", 0),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
        )
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate response with token streaming."""
        client = self._ensure_async_client()
        
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(messages),
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.HTTPError as e:
            logger.error("ollama_stream_failed", error=str(e))
            raise ModelInferenceError(
                model=self.config.model,
                operation="generate_stream",
                reason=str(e),
            )
    
    async def generate_stream_sentences(
        self,
        messages: list[LLMMessage],
        on_sentence: Callable[[str], None],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Stream tokens and call on_sentence for each complete sentence.
        
        This enables TTS to start speaking while LLM is still generating.
        
        Args:
            messages: Conversation history
            on_sentence: Callback for each sentence
            max_tokens: Max tokens to generate
        
        Returns:
            Full response text
        """
        buffer = ""
        full_response = ""
        sentence_count = 0
        
        # Sentence endings
        endings = (".", "!", "?", "。", "！", "？")
        # Minimum chars before checking for sentence end
        min_sentence_len = 20
        
        async for token in self.generate_stream(messages, max_tokens):
            buffer += token
            full_response += token
            
            # Check for sentence boundary
            if len(buffer) >= min_sentence_len:
                for i, char in enumerate(buffer):
                    if char in endings:
                        # Check it's not a decimal or abbreviation
                        if i + 1 < len(buffer) and buffer[i + 1] not in " \n":
                            continue
                        
                        sentence = buffer[:i + 1].strip()
                        if sentence:
                            sentence_count += 1
                            logger.debug(
                                "sentence_ready",
                                num=sentence_count,
                                length=len(sentence),
                            )
                            on_sentence(sentence)
                        
                        buffer = buffer[i + 1:].lstrip()
                        break
        
        # Send any remaining text
        if buffer.strip():
            on_sentence(buffer.strip())
        
        return full_response
    
    def __del__(self) -> None:
        """Cleanup clients."""
        if hasattr(self, "client"):
            self.client.close()
