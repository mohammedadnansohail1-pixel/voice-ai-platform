#!/usr/bin/env python3
"""Test healthcare text WebSocket endpoint."""
import asyncio
import json
import websockets


async def test_healthcare():
    uri = "ws://localhost:8000/healthcare/ws/text"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        # Receive greeting
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"\n🤖 Agent: {data['text']}")
        print(f"   Stage: {data['stage']}, Slots: {data['slots']}\n")
        
        # Test conversation
        test_messages = [
            "I need to schedule an appointment for my headaches",
            "Thursday works for me",
            "9am please",
            "Yes, book it",
        ]
        
        for user_input in test_messages:
            print(f"👤 You: {user_input}")
            
            await ws.send(json.dumps({
                "type": "text",
                "text": user_input,
            }))
            
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"🤖 Agent: {data['text']}")
            print(f"   Stage: {data['stage']}, Slots: {data['slots']}\n")
            
            if data.get('type') == 'complete':
                print(f"✅ Appointment complete!")
                print(f"   Confirmation: {data.get('confirmation')}")
                break
            
            # Check for completion message
            if 'confirmation' in data.get('text', '').lower():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    if data.get('type') == 'complete':
                        print(f"✅ Appointment complete!")
                        print(f"   Confirmation: {data.get('confirmation')}")
                except:
                    pass
                break
        
        print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_healthcare())
