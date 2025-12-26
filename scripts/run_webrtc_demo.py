#!/usr/bin/env python3
"""
WebRTC demo with contextual prompting (practical approach).
"""
import sys
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web
import json
import base64
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("webrtc_demo")

from voice_platform.core.config import load_config
from voice_platform.vad.silero import SileroVAD
from voice_platform.asr.whisper import WhisperASR
from voice_platform.tts.kokoro import KokoroTTS
from voice_platform.llm.ollama import OllamaLLM
from voice_platform.agent import ToolCallingAgent


# Contextual prompts per state (this is what production systems do)
STATE_PROMPTS = {
    "collecting_consent": "Patient consent. Patient says yes, no, agree, consent, okay.",
    "collecting_name": "Patient stating their full name. Names like John Smith, Mohammed Ali, Sarah Johnson.",
    "collecting_dob": "Patient stating date of birth. Dates like March 15 1985, January 1 1990, 03/15/1985.",
    "collecting_phone": "Patient stating phone number. 10 digit US phone numbers like 555-123-4567.",
    "collecting_reason": "Medical appointment. Patient describing symptoms: toothache, headache, back pain, checkup.",
    "collecting_day": "Selecting appointment day: Monday, Tuesday, Wednesday, Thursday, Friday.",
    "confirming_day": "Confirming day. Patient says yes, no, correct, or a day name.",
    "collecting_time": "Selecting time: 9 AM, 10 AM, 2 PM, 3 PM, 4 PM.",
    "confirming": "Confirming appointment. Patient says yes, no, correct, cancel.",
}


class AppState:
    pass


async def create_app():
    app = web.Application()
    
    logger.info("Loading models...")
    config = load_config("configs/base.yaml")
    
    app_state = AppState()
    app_state.config = config
    app_state.vad = SileroVAD(config.vad)
    app_state.asr = WhisperASR(config.asr)  # Back to standard ASR with VAD
    app_state.tts = KokoroTTS(config.tts)
    app_state.llm = OllamaLLM(config.llm)
    
    logger.info("Models loaded!")
    
    async def index_handler(request):
        html_path = Path(__file__).parent.parent / "web" / "index.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="index.html not found", status=404)
    
    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        agent = None
        audio_buffer = []
        
        logger.info("WebSocket connected")
        
        try:
            agent = ToolCallingAgent(
                llm=app_state.llm,
                clinic_name="Sunrise Medical",
            )
            
            greeting = agent.start()
            await ws.send_json({'type': 'transcript', 'role': 'agent', 'text': greeting})
            
            tts_result = app_state.tts.synthesize(greeting)
            audio_b64 = base64.b64encode(
                (tts_result.audio_data * 32767).astype('<i2').tobytes()
            ).decode()
            await ws.send_json({
                'type': 'audio', 'data': audio_b64,
                'sample_rate': tts_result.sample_rate
            })
            
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data['type'] == 'audio':
                        audio_bytes = base64.b64decode(data['data'])
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        src_rate = data.get('sample_rate', 16000)
                        if src_rate != 16000:
                            ratio = 16000 / src_rate
                            new_len = int(len(audio) * ratio)
                            indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
                            audio = audio[indices]
                        
                        audio_buffer.append(audio)
                        
                    elif data['type'] == 'end_speech':
                        if audio_buffer:
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                            
                            # Check audio level
                            max_level = np.max(np.abs(full_audio))
                            if max_level < 0.01:
                                logger.info("Audio too quiet, skipping")
                                continue
                            
                            current_state = agent.context.state.value
                            
                            # Set contextual prompt for this state
                            prompt = STATE_PROMPTS.get(current_state, "")
                            app_state.asr.set_context(
                                state=current_state,
                                expected_entities=prompt.split(": ")[-1].replace(".", "").split(", ") if ": " in prompt else []
                            )
                            
                            # Transcribe with VAD and all safeguards
                            result = app_state.asr.transcribe(full_audio)
                            user_text = result.text.strip()
                            
                            logger.info(f"ASR [{current_state}]: '{user_text}'")
                            
                            if user_text:
                                await ws.send_json({
                                    'type': 'transcript',
                                    'role': 'user',
                                    'text': user_text,
                                })
                                
                                response = agent.process(user_text)
                                
                                logger.info(f"Agent [{agent.context.state.value}]: {response.text[:60]}...")
                                
                                await ws.send_json({
                                    'type': 'transcript',
                                    'role': 'agent',
                                    'text': response.text
                                })
                                
                                tts_result = app_state.tts.synthesize(response.text)
                                audio_b64 = base64.b64encode(
                                    (tts_result.audio_data * 32767).astype('<i2').tobytes()
                                ).decode()
                                await ws.send_json({
                                    'type': 'audio',
                                    'data': audio_b64,
                                    'sample_rate': tts_result.sample_rate
                                })
                                
                                if response.ended:
                                    await ws.send_json({
                                        'type': 'ended',
                                        'booking': response.booking
                                    })
                            else:
                                logger.info("No speech detected")
                                    
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    
        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
        finally:
            logger.info("WebSocket closed")
            
        return ws
    
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', ws_handler)
    
    return app


def main():
    print("=" * 60)
    print("WebRTC Demo - Contextual Prompting")
    print("=" * 60)
    print()
    print("Using standard Whisper with VAD + contextual prompts.")
    print("Open http://localhost:8080 in your browser")
    print()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(create_app())
    web.run_app(app, host='0.0.0.0', port=8080)


if __name__ == "__main__":
    main()
