"""Component registry pattern for plug-and-play backends."""
from typing import Dict, Type, TypeVar, Generic, Any

T = TypeVar('T')


class Registry(Generic[T]):
    """
    Generic registry for component types.
    
    Usage:
        asr_registry = Registry[ASRBackend]("ASR")
        
        @asr_registry.register("whisper")
        class WhisperASR(ASRBackend):
            ...
        
        # Later
        asr = asr_registry.create("whisper", model="large-v3")
    """
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, key: str):
        """Decorator to register a component class."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._registry[key] = cls
            return cls
        return decorator
    
    def get(self, key: str) -> Type[T]:
        """Get component class by key."""
        if key not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Unknown {self.name}: '{key}'. Available: {available}")
        return self._registry[key]
    
    def create(self, key: str, **kwargs: Any) -> T:
        """Create component instance."""
        cls = self.get(key)
        return cls(**kwargs)
    
    def list_available(self) -> list[str]:
        """List all registered component keys."""
        return list(self._registry.keys())
    
    def __contains__(self, key: str) -> bool:
        return key in self._registry
