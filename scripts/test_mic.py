#!/usr/bin/env python3
"""Test microphone audio quality."""
import sys
sys.path.insert(0, "src")

import numpy as np
import wave
import tempfile

print("Recording 3 seconds of audio...")
print("Say 'Thursday' clearly into your microphone")
print()

try:
    import sounddevice as sd
    
    duration = 3  # seconds
    sample_rate = 16000
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    
    audio = audio.flatten()
    
    # Check audio level
    max_level = np.max(np.abs(audio))
    rms_level = np.sqrt(np.mean(audio**2))
    
    print(f"Max level: {max_level:.4f} (should be > 0.1)")
    print(f"RMS level: {rms_level:.4f} (should be > 0.01)")
    
    if max_level < 0.05:
        print("⚠️  Audio too quiet! Check microphone.")
    elif max_level > 0.9:
        print("⚠️  Audio clipping! Mic too loud.")
    else:
        print("✓ Audio level OK")
    
    # Test ASR
    print()
    print("Testing ASR...")
    
    from voice_platform.asr.whisper import WhisperASR
    from voice_platform.core.config import ASRConfig
    
    asr = WhisperASR(ASRConfig(model="large-v3"))
    asr.set_context(state="collecting_day", expected_entities=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    
    result = asr.transcribe(audio)
    print(f"Transcribed: '{result.text}'")
    
    if "thursday" in result.text.lower():
        print("✓ ASR working correctly!")
    else:
        print("✗ ASR did not recognize 'Thursday'")
        
except ImportError:
    print("Install sounddevice: pip install sounddevice")
except Exception as e:
    print(f"Error: {e}")
