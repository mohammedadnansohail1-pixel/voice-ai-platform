"""Pipeline component registries."""
from ..core.registry import Registry
from .base import ASRBackend, TTSBackend, VADBackend, LLMBackend

# Create registries for each component type
asr_registry = Registry[ASRBackend]("ASR backend")
tts_registry = Registry[TTSBackend]("TTS backend")
vad_registry = Registry[VADBackend]("VAD backend")
llm_registry = Registry[LLMBackend]("LLM backend")
