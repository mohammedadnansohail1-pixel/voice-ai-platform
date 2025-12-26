#!/usr/bin/env python3
"""Test foundation components."""

import sys
sys.path.insert(0, "src")

from voice_platform.core import (
    Config,
    load_config,
    Registry,
    vad_registry,
    asr_registry,
    SessionContext,
    SessionState,
    AudioChunk,
    Transcript,
    VoicePlatformError,
    ModelNotFoundError,
)
from voice_platform.logging import setup_logging, get_logger, AuditLogger, AuditEvent

def test_config():
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    # Load from YAML
    config = load_config("configs/base.yaml")
    print(f"✓ Loaded config for tenant: {config.tenant.name}")
    print(f"  ASR: {config.asr.backend} / {config.asr.model}")
    print(f"  TTS: {config.tts.backend} / {config.tts.voice}")
    print(f"  LLM: {config.llm.provider} / {config.llm.model}")
    print(f"  VAD: {config.vad.backend} (threshold={config.vad.threshold})")
    
    # Test defaults
    default_config = Config()
    print(f"✓ Default config works: {default_config.tenant.id}")
    
    return config

def test_registry():
    print("\n" + "=" * 60)
    print("TEST 2: Component Registry")
    print("=" * 60)
    
    # Test registry creation
    print(f"✓ Registries available: {Registry.list_registries()}")
    
    # Test registration decorator
    @vad_registry.register("test_vad")
    class TestVAD:
        def __init__(self, config=None):
            self.config = config
        def detect(self, audio):
            return True
    
    print(f"✓ Registered test_vad: {vad_registry.list_backends()}")
    
    # Test retrieval
    vad_class = vad_registry.get("test_vad")
    vad = vad_class()
    print(f"✓ Retrieved and instantiated: {vad.__class__.__name__}")
    
    # Test error handling
    try:
        vad_registry.get("nonexistent")
    except ModelNotFoundError as e:
        print(f"✓ ModelNotFoundError raised correctly: {e.code}")

def test_types():
    print("\n" + "=" * 60)
    print("TEST 3: Type Definitions")
    print("=" * 60)
    
    # Test AudioChunk
    chunk = AudioChunk(data=b"\x00" * 3200, sample_rate=16000)
    print(f"✓ AudioChunk: {chunk.duration_ms:.1f}ms")
    
    # Test Transcript
    transcript = Transcript.from_text("Hello, how can I help?", language="en")
    print(f"✓ Transcript: '{transcript.text}' (lang={transcript.language})")
    
    # Test SessionContext
    session = SessionContext()
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there!")
    print(f"✓ Session: {session.session_id[:8]}... ({len(session.messages)} messages)")
    print(f"  State: {session.state.value}")
    print(f"  History: {[m.role for m in session.get_conversation_history()]}")

def test_logging(config):
    print("\n" + "=" * 60)
    print("TEST 4: Logging & Audit")
    print("=" * 60)
    
    # Setup logging
    setup_logging(config.logging)
    logger = get_logger("test")
    logger.info("test_message", component="foundation", status="ok")
    print("✓ Structured logging works")
    
    # Test audit logger
    audit = AuditLogger(
        enabled=config.logging.audit_enabled,
        redact_phi=config.logging.audit_redact_phi,
        audit_path=config.logging.audit_path,
    )
    
    audit.session_start("test-session-123", channel="local_mic")
    audit.transcript("test-session-123", "My SSN is 123-45-6789")
    audit.session_end("test-session-123", duration_s=5.5)
    print(f"✓ Audit logger works (PHI redaction={audit.redact_phi_enabled})")

def test_exceptions():
    print("\n" + "=" * 60)
    print("TEST 5: Exception Hierarchy")
    print("=" * 60)
    
    from voice_platform.core import get_http_status
    
    errors = [
        ModelNotFoundError("asr", "whisper"),
        VoicePlatformError("Generic error", "CUSTOM_ERROR"),
    ]
    
    for e in errors:
        status = get_http_status(e)
        print(f"✓ {e.__class__.__name__}: HTTP {status} - {e.code}")

def main():
    print("\n" + "=" * 60)
    print("VOICE AI PLATFORM - FOUNDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        config = test_config()
        test_registry()
        test_types()
        test_logging(config)
        test_exceptions()
        
        print("\n" + "=" * 60)
        print("✅ ALL FOUNDATION TESTS PASSED")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
