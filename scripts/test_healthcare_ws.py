#!/usr/bin/env python3
"""Test healthcare WebSocket endpoint with text simulation."""
import asyncio
import json
import base64
import numpy as np

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    exit(1)


async def test_healthcare_text():
    """Test healthcare endpoint with simulated audio (text mode)."""
    uri = "ws://localhost:8000/healthcare/ws/voice"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("Connected!")
        
        # Receive greeting
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"  <- {data['type']}: {data.get('text', data.get('status', ''))[:80]}")
            
            if data['type'] == 'response':
                break
        
        # Wait for audio to finish
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data['type'] == 'status' and data['status'] == 'listening':
                break
        
        print("\nReady for conversation. Type your messages:")
        print("(This is text simulation - real audio would be base64 PCM)\n")
        
        # Simulate conversation
        test_messages = [
            "I need to schedule an appointment for my headaches",
            "Thursday works",
            "9am please",
            "Yes, book it",
        ]
        
        for user_input in test_messages:
            print(f"  -> Simulating: '{user_input}'")
            
            # In real usage, you'd send actual audio
            # For testing, we'll create a tiny fake audio packet
            # The ASR won't understand it, but we can see the flow
            fake_audio = np.zeros(1600, dtype=np.int16)  # 0.1s of silence at 16kHz
            audio_b64 = base64.b64encode(fake_audio.tobytes()).decode()
            
            await ws.send(json.dumps({
                "type": "audio",
                "data": audio_b64,
            }))
            
            # Note: This won't work properly because we're sending silence
            # Real test needs actual audio or a text endpoint
            await asyncio.sleep(0.5)
        
        print("\nNote: Text simulation doesn't work with audio endpoint.")
        print("Use the browser with microphone or MicroSIP for real testing.")


async def main():
    try:
        await test_healthcare_text()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
