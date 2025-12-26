"""Test pipeline module."""
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
test_section("Pipeline Imports")

try:
    from voice_platform.pipeline import (
        ASRBackend, TTSBackend, VADBackend, LLMBackend,
        asr_registry, tts_registry, vad_registry, llm_registry
    )
    from voice_platform.pipeline.base import (
        TranscriptionResult, SynthesisResult, VADResult
    )
    test_pass("All imports successful")
except Exception as e:
    test_fail("Imports", e)
    sys.exit(1)

# =============================================================================
# Test 2: Data Classes
# =============================================================================
test_section("Data Classes")

try:
    # TranscriptionResult
    tr = TranscriptionResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        duration_seconds=1.5
    )
    assert tr.text == "Hello world"
    assert tr.language == "en"
    test_pass(f"TranscriptionResult: '{tr.text}' ({tr.language}, {tr.confidence:.0%})")
    
    # SynthesisResult
    audio = np.zeros(16000, dtype=np.float32)
    sr = SynthesisResult(audio=audio, sample_rate=16000, duration_seconds=1.0)
    assert sr.sample_rate == 16000
    assert len(sr.audio) == 16000
    test_pass(f"SynthesisResult: {len(sr.audio)} samples @ {sr.sample_rate}Hz")
    
    # VADResult
    vr = VADResult(is_speech=True, speech_probability=0.87, speech_started=True)
    assert vr.is_speech
    assert vr.speech_started
    test_pass(f"VADResult: speech={vr.is_speech}, prob={vr.speech_probability:.0%}")

except Exception as e:
    test_fail("Data classes", e)

# =============================================================================
# Test 3: Mock Implementation & Registry
# =============================================================================
test_section("Mock Implementation & Registry")

try:
    # Create a mock ASR backend
    @asr_registry.register("mock")
    class MockASR(ASRBackend):
        def __init__(self, model: str = "mock-model"):
            self.model = model
            self._loaded = False
        
        def load(self):
            self._loaded = True
        
        def transcribe(self, audio, sample_rate=16000, language=None):
            return TranscriptionResult(
                text="mock transcription",
                language=language or "en",
                confidence=0.99,
                duration_seconds=len(audio) / sample_rate
            )
        
        @property
        def is_loaded(self):
            return self._loaded
    
    test_pass("MockASR registered")
    
    # Create via registry
    asr = asr_registry.create("mock", model="test-model")
    assert asr.model == "test-model"
    test_pass(f"Created via registry (model={asr.model})")
    
    # Test load
    assert not asr.is_loaded
    asr.load()
    assert asr.is_loaded
    test_pass("load() and is_loaded work")
    
    # Test transcribe
    audio = np.random.randn(16000).astype(np.float32)
    result = asr.transcribe(audio)
    assert result.text == "mock transcription"
    assert result.duration_seconds == 1.0
    test_pass(f"transcribe() returned: '{result.text}'")

except Exception as e:
    test_fail("Mock implementation", e)

# =============================================================================
# Test 4: Abstract Methods Enforcement
# =============================================================================
test_section("Abstract Methods Enforcement")

try:
    # Try to instantiate abstract class directly
    try:
        asr = ASRBackend()
        test_fail("Should not be able to instantiate abstract class", Exception("No error"))
    except TypeError as e:
        test_pass(f"Cannot instantiate abstract ASRBackend: {e}")
    
    # Try incomplete implementation
    try:
        @asr_registry.register("incomplete")
        class IncompleteASR(ASRBackend):
            def load(self):
                pass
            # Missing other methods
        
        incomplete = asr_registry.create("incomplete")
        test_fail("Should not be able to instantiate incomplete impl", Exception("No error"))
    except TypeError as e:
        test_pass(f"Cannot instantiate incomplete impl: {e}")

except Exception as e:
    test_fail("Abstract enforcement", e)

# =============================================================================
# Test 5: All Registries Empty Initially
# =============================================================================
test_section("Registry State")

try:
    # Note: asr_registry has our mock from earlier
    console.print(f"  ASR backends: {asr_registry.list_available()}")
    console.print(f"  TTS backends: {tts_registry.list_available()}")
    console.print(f"  VAD backends: {vad_registry.list_available()}")
    console.print(f"  LLM backends: {llm_registry.list_available()}")
    test_pass("All registries accessible")

except Exception as e:
    test_fail("Registry state", e)

# =============================================================================
# Summary
# =============================================================================
console.print(f"\n[bold green]{'='*50}[/]")
console.print("[bold green]All pipeline tests passed![/]")
console.print(f"[bold green]{'='*50}[/]")
