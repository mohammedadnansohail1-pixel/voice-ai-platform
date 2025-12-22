"""
Voice AI Platform - Component Registry

Enables pluggable backends for VAD, ASR, TTS, LLM via decorators.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from .exceptions import ModelNotFoundError

T = TypeVar("T")


class Registry:
    """
    Generic registry for pluggable components.
    
    Usage:
        vad_registry = Registry("vad")
        
        @vad_registry.register("silero")
        class SileroVAD:
            ...
        
        # Later
        vad_class = vad_registry.get("silero")
        vad = vad_class(config)
    """
    
    _registries: Dict[str, "Registry"] = {}
    
    def __init__(self, category: str) -> None:
        self.category = category
        self._backends: Dict[str, Type[Any]] = {}
        Registry._registries[category] = self
    
    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a backend."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._backends[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type[Any]:
        """Get a registered backend by name."""
        if name not in self._backends:
            raise ModelNotFoundError(
                category=self.category,
                backend=name,
            )
        return self._backends[name]
    
    def list_backends(self) -> list[str]:
        """List all registered backends."""
        return list(self._backends.keys())
    
    def has(self, name: str) -> bool:
        """Check if a backend is registered."""
        return name in self._backends
    
    @classmethod
    def get_registry(cls, category: str) -> Optional["Registry"]:
        """Get a registry by category."""
        return cls._registries.get(category)
    
    @classmethod
    def list_registries(cls) -> list[str]:
        """List all registry categories."""
        return list(cls._registries.keys())


# Pre-defined registries for each component type
vad_registry = Registry("vad")
asr_registry = Registry("asr")
tts_registry = Registry("tts")
llm_registry = Registry("llm")
channel_registry = Registry("channel")


def get_component(category: str, backend: str, config: Any = None) -> Any:
    """
    Factory function to instantiate a component.
    
    Args:
        category: Component category (vad, asr, tts, llm, channel)
        backend: Backend name (silero, whisper, kokoro, etc.)
        config: Configuration to pass to the backend
    
    Returns:
        Instantiated component
    """
    registry = Registry.get_registry(category)
    if registry is None:
        raise ModelNotFoundError(category=category, backend=None)
    
    backend_class = registry.get(backend)
    
    if config is not None:
        return backend_class(config)
    return backend_class()
