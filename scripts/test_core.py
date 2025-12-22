"""Test core modules."""
import sys
sys.path.insert(0, 'src')

from rich.console import Console
from rich.table import Table

console = Console()

def test_section(name: str):
    console.print(f"\n[bold cyan]{'='*50}[/]")
    console.print(f"[bold cyan]Testing: {name}[/]")
    console.print(f"[bold cyan]{'='*50}[/]")

def test_pass(msg: str):
    console.print(f"  [green]✓[/] {msg}")

def test_fail(msg: str, error: Exception):
    console.print(f"  [red]✗[/] {msg}")
    console.print(f"    [red]{type(error).__name__}: {error}[/]")

# =============================================================================
# Test 1: Exceptions
# =============================================================================
test_section("Exceptions")

try:
    from voice_platform.core.exceptions import (
        VoicePlatformError, ConfigError, ChannelError, PipelineError, IntegrationError
    )
    
    # Test exception hierarchy
    assert issubclass(ConfigError, VoicePlatformError)
    assert issubclass(ChannelError, VoicePlatformError)
    assert issubclass(PipelineError, VoicePlatformError)
    assert issubclass(IntegrationError, VoicePlatformError)
    test_pass("Exception hierarchy correct")
    
    # Test raising and catching
    try:
        raise ConfigError("Test config error")
    except VoicePlatformError as e:
        test_pass(f"ConfigError caught as VoicePlatformError: {e}")
    
except Exception as e:
    test_fail("Exceptions module", e)

# =============================================================================
# Test 2: Registry
# =============================================================================
test_section("Registry")

try:
    from voice_platform.core.registry import Registry
    
    # Create a test registry
    class Backend:
        pass
    
    registry = Registry[Backend]("TestBackend")
    test_pass("Registry created")
    
    # Register a component
    @registry.register("mock")
    class MockBackend(Backend):
        def __init__(self, value: int = 0):
            self.value = value
    
    test_pass("Component registered with decorator")
    
    # List available
    available = registry.list_available()
    assert "mock" in available
    test_pass(f"list_available(): {available}")
    
    # Check contains
    assert "mock" in registry
    assert "nonexistent" not in registry
    test_pass("__contains__ works")
    
    # Get class
    cls = registry.get("mock")
    assert cls == MockBackend
    test_pass("get() returns correct class")
    
    # Create instance
    instance = registry.create("mock", value=42)
    assert isinstance(instance, MockBackend)
    assert instance.value == 42
    test_pass(f"create() with kwargs works (value={instance.value})")
    
    # Test error on unknown key
    try:
        registry.get("unknown")
        test_fail("Should have raised KeyError", Exception("No error raised"))
    except KeyError as e:
        test_pass(f"KeyError on unknown key: {e}")

except Exception as e:
    test_fail("Registry module", e)

# =============================================================================
# Test 3: Config
# =============================================================================
test_section("Config")

try:
    from voice_platform.core.config import (
        Config, load_config, AudioConfig, VADConfig, ASRConfig, 
        LLMConfig, TTSConfig, _deep_merge, _substitute_env_vars
    )
    
    # Test default config
    config = Config()
    assert config.audio.sample_rate == 16000
    assert config.vad.backend == "silero"
    assert config.asr.backend == "whisper"
    test_pass("Default config values correct")
    
    # Test deep merge
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}
    test_pass("Deep merge works correctly")
    
    # Test env var substitution
    import os
    os.environ["TEST_VAR"] = "test_value"
    data = {"key": "${TEST_VAR}", "nested": {"key2": "${TEST_VAR}"}}
    result = _substitute_env_vars(data)
    assert result["key"] == "test_value"
    assert result["nested"]["key2"] == "test_value"
    test_pass("Environment variable substitution works")
    
    # Test loading from file
    config = load_config("configs/base.yaml")
    assert config.tenant.id == "default"
    assert config.asr.model == "large-v3"
    assert config.tts.voice == "af_sarah"
    test_pass(f"Loaded config from YAML (tenant={config.tenant.name})")
    
    # Test loading nonexistent file (should return defaults)
    config = load_config("configs/nonexistent.yaml")
    assert config.audio.sample_rate == 16000
    test_pass("Nonexistent config file returns defaults")

except Exception as e:
    test_fail("Config module", e)

# =============================================================================
# Test 4: Logging
# =============================================================================
test_section("Logging")

try:
    from voice_platform.logging import get_logger, setup_logging, AuditLogger, AuditEvent
    from voice_platform.logging.audit import AuditEventType
    from voice_platform.core.config import LoggingConfig
    import tempfile
    import os
    
    # Setup logging
    setup_logging(LoggingConfig(level="INFO", format="console"))
    test_pass("Logging setup complete")
    
    # Get logger
    logger = get_logger("test")
    logger.info("Test log message", extra_field="test_value")
    test_pass("Logger works with extra fields")
    
    # Test audit event
    event = AuditEvent(
        event_type=AuditEventType.CALL_STARTED,
        tenant_id="test_tenant",
        session_id="session_123",
        caller_id="+15551234567",
        data={"channel": "twilio", "ssn": "123-45-6789"}
    )
    
    # Test PHI redaction
    redacted = event.to_dict(redact_phi=True)
    assert redacted["caller_id"] == "***-***-4567"
    assert redacted["data"]["ssn"] == "[REDACTED]"
    assert redacted["data"]["channel"] == "twilio"  # Not PHI
    test_pass("PHI redaction works (phone, ssn)")
    
    # Test without redaction
    unredacted = event.to_dict(redact_phi=False)
    assert unredacted["caller_id"] == "+15551234567"
    assert unredacted["data"]["ssn"] == "123-45-6789"
    test_pass("Unredacted mode preserves data")
    
    # Test audit logger file writing
    with tempfile.TemporaryDirectory() as tmpdir:
        audit = AuditLogger(path=tmpdir, redact_phi=True)
        audit.log_call_start("tenant1", "session1", "+15559876543", "webrtc")
        audit.log_call_end("tenant1", "session1", 45.5, "completed")
        
        # Check file was created
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith("audit_")
        test_pass(f"Audit file created: {files[0]}")
        
        # Check content
        with open(os.path.join(tmpdir, files[0])) as f:
            lines = f.readlines()
        assert len(lines) == 2
        test_pass(f"Audit file has {len(lines)} events")

except Exception as e:
    test_fail("Logging module", e)

# =============================================================================
# Summary
# =============================================================================
console.print(f"\n[bold green]{'='*50}[/]")
console.print("[bold green]All core module tests passed![/]")
console.print(f"[bold green]{'='*50}[/]")
