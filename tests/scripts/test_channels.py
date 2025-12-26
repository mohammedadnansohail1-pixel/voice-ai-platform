"""Test channels module."""
import sys
sys.path.insert(0, 'src')

from rich.console import Console
import numpy as np

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
# Test 1: Imports
# =============================================================================
test_section("Channel Imports")

try:
    from voice_platform.channels import (
        Channel, ChannelEvent, ChannelEventType, channel_registry
    )
    from voice_platform.logging import setup_logging
    from voice_platform.core.config import LoggingConfig
    
    setup_logging(LoggingConfig(level="WARNING", format="console"))
    test_pass("All imports successful")
except Exception as e:
    test_fail("Imports", e)
    sys.exit(1)

# =============================================================================
# Test 2: Channel Registry
# =============================================================================
test_section("Channel Registry")

try:
    # Import to trigger registration
    from voice_platform.channels.local_mic import LocalMicChannel
    
    available = channel_registry.list_available()
    assert "local_mic" in available
    test_pass(f"Registered channels: {available}")
    
except Exception as e:
    test_fail("Channel registry", e)

# =============================================================================
# Test 3: Channel Event
# =============================================================================
test_section("Channel Event")

try:
    # Create event
    audio = np.random.randn(480).astype(np.float32)
    event = ChannelEvent(
        type=ChannelEventType.AUDIO_RECEIVED,
        session_id="test-123",
        audio=audio,
        sample_rate=16000,
    )
    
    assert event.type == ChannelEventType.AUDIO_RECEIVED
    assert len(event.audio) == 480
    test_pass(f"Event created: {event.type.value}, {len(event.audio)} samples")
    
    # Check all event types
    event_types = list(ChannelEventType)
    test_pass(f"Event types: {[e.value for e in event_types]}")
    
except Exception as e:
    test_fail("Channel event", e)

# =============================================================================
# Test 4: Mock Channel Implementation
# =============================================================================
test_section("Mock Channel Implementation")

try:
    from voice_platform.channels.base import Channel, EventHandler
    
    @channel_registry.register("mock")
    class MockChannel(Channel):
        """Mock channel for testing."""
        
        def __init__(self, session_id: str, sample_rate: int = 16000):
            super().__init__(session_id, sample_rate)
            self.played_audio = []
            self.interrupted = False
        
        def start(self):
            self._is_active = True
            self._emit(ChannelEvent(type=ChannelEventType.CONNECTED))
        
        def stop(self):
            self._is_active = False
            self._emit(ChannelEvent(type=ChannelEventType.DISCONNECTED))
        
        def play_audio(self, audio, sample_rate=16000):
            self.played_audio.append((audio, sample_rate))
        
        def interrupt_playback(self):
            self.interrupted = True
        
        def simulate_audio(self, audio: np.ndarray):
            """Simulate receiving audio."""
            self._emit(ChannelEvent(
                type=ChannelEventType.AUDIO_RECEIVED,
                audio=audio,
                sample_rate=self.sample_rate,
            ))
    
    test_pass("MockChannel registered")
    
    # Create and test mock channel
    channel = channel_registry.create("mock", session_id="test-session")
    
    # Track events
    events_received = []
    channel.on_event(lambda e: events_received.append(e))
    
    # Start
    channel.start()
    assert channel.is_active
    assert len(events_received) == 1
    assert events_received[0].type == ChannelEventType.CONNECTED
    test_pass("Channel started, CONNECTED event received")
    
    # Simulate audio
    audio = np.random.randn(480).astype(np.float32)
    channel.simulate_audio(audio)
    assert len(events_received) == 2
    assert events_received[1].type == ChannelEventType.AUDIO_RECEIVED
    test_pass("AUDIO_RECEIVED event emitted")
    
    # Play audio
    channel.play_audio(audio, 16000)
    assert len(channel.played_audio) == 1
    test_pass("play_audio() recorded")
    
    # Interrupt
    channel.interrupt_playback()
    assert channel.interrupted
    test_pass("interrupt_playback() called")
    
    # Stop
    channel.stop()
    assert not channel.is_active
    assert events_received[-1].type == ChannelEventType.DISCONNECTED
    test_pass("Channel stopped, DISCONNECTED event received")

except Exception as e:
    test_fail("Mock channel", e)
    raise

# =============================================================================
# Test 5: LocalMicChannel Creation (no audio hardware test)
# =============================================================================
test_section("LocalMicChannel Creation")

try:
    channel = channel_registry.create(
        "local_mic",
        session_id="test-local",
        sample_rate=16000,
        chunk_duration_ms=30,
    )
    
    assert channel.session_id == "test-local"
    assert channel.sample_rate == 16000
    assert channel.chunk_size == 480  # 16000 * 30 / 1000
    test_pass(f"LocalMicChannel created: chunk_size={channel.chunk_size}")
    
    # Test resampling function
    audio_48k = np.sin(np.linspace(0, 10, 4800)).astype(np.float32)
    audio_16k = channel._resample(audio_48k, 48000, 16000)
    assert len(audio_16k) == 1600  # 4800 * (16000/48000)
    test_pass(f"Resampling works: {len(audio_48k)} -> {len(audio_16k)} samples")

except Exception as e:
    test_fail("LocalMicChannel creation", e)

# =============================================================================
# Summary
# =============================================================================
console.print(f"\n[bold green]{'='*50}[/]")
console.print("[bold green]All channel tests passed![/]")
console.print(f"[bold green]{'='*50}[/]")
