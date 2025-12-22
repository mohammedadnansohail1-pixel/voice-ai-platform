"""Ollama LLM implementation."""
import time
from typing import AsyncIterator, Optional

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
    Ollama LLM backend for local inference.
    
    Supports streaming responses for low-latency TTS handoff.
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
        """Generate response with streaming."""
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
                    
                    import json
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
    
    def __del__(self) -> None:
        """Cleanup clients."""
        if hasattr(self, "client"):
            self.client.close()
