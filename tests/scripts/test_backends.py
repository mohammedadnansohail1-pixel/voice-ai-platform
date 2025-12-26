"""Test real ASR and TTS backends."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from rich.console import Console

console = Console()

def test_section(name: str):
    console.print(f"\n[bold cyan]{'='*50}[/]")
    console.print(f"[bold cyan]{name}[/]")
    console.print(f"[bold cyan]{'='*50}[/]")

# Setup logging
from voice_platform.logging import setup_logging
from voice_platform.core.config import LoggingConfig
setup_logging(LoggingConfig(level="INFO", format="console"))

# =============================================================================
# Test Whisper ASR
# =============================================================================
test_section("Testing Whisper ASR")

try:
    from voice_platform.pipeline.asr_whisper import WhisperASR
    
    console.print("Creating WhisperASR (large-v3, cuda)...")
    asr = WhisperASR(model="large-v3", device="cuda", compute_type="float16")
    
    console.print("Loading model (this may take a moment)...")
    asr.load()
    console.print("[green]✓ Model loaded[/]")
    
    # Create test audio (1 second of silence + sine wave)
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    console.print("Transcribing test audio...")
    result = asr.transcribe(audio, sample_rate=sr)
    console.print(f"[green]✓ Transcription:[/] '{result.text}' (lang={result.language})")
    
except Exception as e:
    console.print(f"[red]✗ ASR Error: {e}[/]")
    raise

# =============================================================================
# Test Kokoro TTS
# =============================================================================
test_section("Testing Kokoro TTS")

try:
    from voice_platform.pipeline.tts_kokoro import KokoroTTS
    
    console.print("Creating KokoroTTS (af_heart, cuda)...")
    tts = KokoroTTS(voice="af_heart", device="cuda")
    
    console.print("Loading model...")
    tts.load()
    console.print("[green]✓ Model loaded[/]")
    
    console.print("Synthesizing speech...")
    result = tts.synthesize("Hello! This is a test of the voice AI platform.")
    console.print(f"[green]✓ Synthesized:[/] {result.duration_seconds:.2f}s, {len(result.audio)} samples @ {result.sample_rate}Hz")
    
    # Save to file for verification
    import soundfile as sf
    sf.write("/tmp/test_tts.wav", result.audio, result.sample_rate)
    console.print("[green]✓ Saved to /tmp/test_tts.wav[/]")
    
except Exception as e:
    console.print(f"[red]✗ TTS Error: {e}[/]")
    raise

console.print(f"\n[bold green]{'='*50}[/]")
console.print("[bold green]All backend tests passed![/]")
console.print(f"[bold green]{'='*50}[/]")
