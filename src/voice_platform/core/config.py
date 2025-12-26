"""
Voice AI Platform - Hierarchical Configuration System

Supports YAML-based configuration with environment variable overrides.
"""

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


class TelephonyConfig(BaseModel):
    """Telephony/Twilio settings."""
    enabled: bool = False
    provider: str = "twilio"
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    phone_number: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging settings."""
    level: str = "INFO"
    format: str = "console"  # console or json
    audit_enabled: bool = True
    audit_redact_phi: bool = True
    audit_path: str = "logs/audit"


class MetricsConfig(BaseModel):
    """Metrics/monitoring settings."""
    enabled: bool = True
    port: int = 9090




class EventBusConfig(BaseModel):
    """Event bus settings for multi-agent coordination."""
    backend: str = "redis"  # redis, kafka (future)
    redis_url: str = "redis://:Adsohnan213!456@localhost:6379/0"
    channel_prefix: str = "voice_ai"  # Namespace: {prefix}:{tenant}:{event_type}
    
    # Redis-specific
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0
    
    # Future Kafka settings (ignored when backend=redis)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "voice_ai_agents"


class CheckpointConfig(BaseModel):
    """Checkpoint/recovery settings for long-running agent calls."""
    backend: str = "postgresql"  # postgresql, sqlite (dev only)
    
    # PostgreSQL connection
    dsn: str = "postgresql://postgres:postgres@localhost:5432/voice_ai"
    
    # SQLite fallback (for local dev without postgres)
    sqlite_path: str = "data/checkpoints.db"
    
    # Retention
    retention_hours: int = 72  # Delete checkpoints older than this
    
    # Checkpointing behavior
    auto_checkpoint_interval_s: int = 60  # For long holds (payer agent)
    checkpoint_on_state_change: bool = True  # Checkpoint at each state transition



class EventBusConfig(BaseModel):
    """Event bus settings for multi-agent coordination."""
    backend: str = "redis"  # redis, kafka (future)
    redis_url: str = "redis://:Adsohnan213!456@localhost:6379/0"
    channel_prefix: str = "voice_ai"  # Namespace: {prefix}:{tenant}:{event_type}
    
    # Redis-specific
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0
    
    # Future Kafka settings (ignored when backend=redis)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "voice_ai_agents"


class CheckpointConfig(BaseModel):
    """Checkpoint/recovery settings for long-running agent calls."""
    backend: str = "postgresql"  # postgresql, sqlite (dev only)
    
    # PostgreSQL connection
    dsn: str = "postgresql://postgres:postgres@localhost:5432/voice_ai"
    
    # SQLite fallback (for local dev without postgres)
    sqlite_path: str = "data/checkpoints.db"
    
    # Retention
    retention_hours: int = 72  # Delete checkpoints older than this
    
    # Checkpointing behavior
    auto_checkpoint_interval_s: int = 60  # For long holds (payer agent)
    checkpoint_on_state_change: bool = True  # Checkpoint at each state transition

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
    telephony: TelephonyConfig = Field(default_factory=TelephonyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment overrides.
    
    Priority: Environment variables > YAML file > Defaults
    """
    config_data: dict[str, Any] = {}
    
    if path:
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}
    
    # Environment variable overrides (VP_ prefix)
    env_mappings = {
        "VP_ASR_BACKEND": ("asr", "backend"),
        "VP_ASR_MODEL": ("asr", "model"),
        "VP_ASR_DEVICE": ("asr", "device"),
        "VP_LLM_PROVIDER": ("llm", "provider"),
        "VP_LLM_MODEL": ("llm", "model"),
        "VP_LLM_BASE_URL": ("llm", "base_url"),
        "VP_TTS_BACKEND": ("tts", "backend"),
        "VP_TTS_VOICE": ("tts", "voice"),
        "VP_LOG_LEVEL": ("logging", "level"),
        "VP_TELEPHONY_ACCOUNT_SID": ("telephony", "account_sid"),
        "VP_TELEPHONY_AUTH_TOKEN": ("telephony", "auth_token"),
        "VP_EVENT_BUS_BACKEND": ("event_bus", "backend"),
        "VP_EVENT_BUS_REDIS_URL": ("event_bus", "redis_url"),
        "VP_CHECKPOINT_BACKEND": ("checkpoint", "backend"),
        "VP_CHECKPOINT_DSN": ("checkpoint", "dsn"),
        "VP_EVENT_BUS_BACKEND": ("event_bus", "backend"),
        "VP_EVENT_BUS_REDIS_URL": ("event_bus", "redis_url"),
        "VP_CHECKPOINT_BACKEND": ("checkpoint", "backend"),
        "VP_CHECKPOINT_DSN": ("checkpoint", "dsn"),
    }
    
    for env_var, (section, key) in env_mappings.items():
        value = os.environ.get(env_var)
        if value:
            if section not in config_data:
                config_data[section] = {}
            config_data[section][key] = value
    
    return Config(**config_data)
