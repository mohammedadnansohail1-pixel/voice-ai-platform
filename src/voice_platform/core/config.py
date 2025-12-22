"""Hierarchical configuration system."""
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AudioConfig(BaseModel):
    """Audio settings."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30


class VADConfig(BaseModel):
    """Voice activity detection settings."""
    backend: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    max_speech_s: float = 30.0


class ASRConfig(BaseModel):
    """Speech-to-text settings."""
    backend: str = "whisper"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: Optional[str] = None  # None = auto-detect


class LLMConfig(BaseModel):
    """Language model settings."""
    provider: str = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 200
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief and conversational."


class TTSConfig(BaseModel):
    """Text-to-speech settings."""
    backend: str = "kokoro"
    voice: str = "af_sarah"
    device: str = "cuda"
    
    # Language-specific voices
    voices: dict[str, str] = Field(default_factory=lambda: {
        "en": "af_sarah",
        "ar": "ar_JO-kareem-medium",
        "hi": "hi_IN-pratham-medium",
    })


class BargeInConfig(BaseModel):
    """Barge-in/interruption settings."""
    enabled: bool = True
    grace_period_ms: int = 500
    similarity_threshold: float = 0.5
    energy_threshold: float = 0.03


class ChannelsConfig(BaseModel):
    """Channel settings."""
    enabled: list[str] = Field(default_factory=lambda: ["local_mic"])
    
    # Twilio settings (loaded from env)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging settings."""
    level: str = "INFO"
    format: str = "json"  # json or console
    
    # Audit logging
    audit_enabled: bool = True
    audit_redact_phi: bool = True
    audit_path: str = "logs/audit"


class MetricsConfig(BaseModel):
    """Metrics/observability settings."""
    enabled: bool = True
    port: int = 9090


class TenantConfig(BaseModel):
    """Tenant identification."""
    id: str = "default"
    name: str = "Default Tenant"


class Config(BaseModel):
    """Root configuration."""
    tenant: TenantConfig = Field(default_factory=TenantConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute ${VAR} with environment variables."""
    if isinstance(data, str):
        if data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.environ.get(var_name, "")
        return data
    elif isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    return data


def load_config(
    base_path: str = "configs/base.yaml",
    tenant_id: Optional[str] = None,
) -> Config:
    """
    Load configuration with optional tenant override.
    
    Args:
        base_path: Path to base configuration file
        tenant_id: Optional tenant ID for tenant-specific overrides
        
    Returns:
        Merged configuration
    """
    base_file = Path(base_path)
    
    # Load base config
    if base_file.exists():
        with open(base_file) as f:
            base_data = yaml.safe_load(f) or {}
    else:
        base_data = {}
    
    # Load tenant override if specified
    if tenant_id:
        tenant_file = base_file.parent / "tenants" / f"{tenant_id}.yaml"
        if tenant_file.exists():
            with open(tenant_file) as f:
                tenant_data = yaml.safe_load(f) or {}
            base_data = _deep_merge(base_data, tenant_data)
    
    # Substitute environment variables
    base_data = _substitute_env_vars(base_data)
    
    return Config(**base_data)
