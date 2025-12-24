"""Healthcare voice API routes."""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from ...core.config import Config
from ...core.types import SessionState
from ...logging import get_logger
from ...audio.accumulator import SpeechAccumulator
from ...channels.websocket import WebSocketChannel
from ...healthcare import (
    HealthcareConversationAgent,
    HealthcareConfig,
    load_healthcare_config,
)

logger = get_logger("api.healthcare")

router = APIRouter(prefix="/healthcare", tags=["healthcare"])


@router.get("/config")
async def get_healthcare_config():
    """Get current healthcare configuration."""
    try:
        config = load_healthcare_config("configs/healthcare/clinic.yaml")
        return {
            "clinic_name": config.clinic.name,
            "verification_enabled": config.verification.enabled,
            "available_slots": [
                {"day": day, "time": time} 
                for day, time in config.available_slots
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/voice")
async def healthcare_voice_websocket(websocket: WebSocket):
    """
    Healthcare voice WebSocket endpoint.
    
    Uses HealthcareConversationAgent for appointment scheduling.
    """
    from ...vad import SileroVAD
    from ...asr import WhisperASR
    from ...tts import KokoroTTS
    from ...core.config import load_config
    
    app_state = websocket.app.state.platform
    
    if not app_state.is_loaded:
        await websocket.close(code=1013, reason="Models not loaded")
        return
    
    try:
        healthcare_config = load_healthcare_config("configs/healthcare/clinic.yaml")
    except:
        healthcare_config = load_healthcare_config(None)
    
    agent = HealthcareConversationAgent(
        clinic_name=healthcare_config.clinic.name,
        require_verification=False,
        available_slots=healthcare_config.available_slots,
    )
    
    config = app_state.config
    channel = WebSocketChannel(websocket, config)
    await channel.accept()
    
    accumulator = SpeechAccumulator(config.vad)
    
    app_state.audit.session_start(channel.session.session_id, channel="healthcare_websocket")
    
    try:
        await channel.send_status("ready")
        
        response = agent.start(session_id=channel.session.session_id)
        
        logger.info(
            "healthcare_session_started",
            session_id=channel.session.session_id[:8],
            clinic=healthcare_config.clinic.name,
        )
        
        await websocket.send_json({
            "type": "response",
            "text": response.message,
            "slots": response.slots,
            "stage": response.stage.value,
        })
        
        await channel.send_status("speaking")
        channel.session.is_speaking = True
        
        tts_result = app_state.tts.synthesize(response.message)
        await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
        
        channel.session.is_speaking = False
        await channel.send_status("listening")
        
        async for audio_chunk in channel.receive_audio():
            if channel.session.is_speaking:
                vad_result = app_state.vad.process_chunk(audio_chunk)
                if vad_result.is_speech and config.barge_in.enabled:
                    channel.session.is_speaking = False
                    await channel.send_status("interrupted")
                    accumulator.reset()
                    continue
            
            vad_result = app_state.vad.process_chunk(audio_chunk)
            segment = accumulator.process(audio_chunk, vad_result)
            
            if segment:
                await channel.send_status("processing")
                channel.session.state = SessionState.PROCESSING
                
                transcript = app_state.asr.transcribe(segment.audio)
                
                if transcript.text.strip():
                    user_text = transcript.text.strip()
                    
                    await channel.send_transcript(user_text, is_final=True)
                    logger.info("user_said", text=user_text, session_id=channel.session.session_id[:8])
                    
                    response = agent.process(user_text)
                    
                    logger.info(
                        "agent_response",
                        stage=response.stage.value,
                        slots=list(response.slots.keys()),
                        ended=response.ended,
                    )
                    
                    await websocket.send_json({
                        "type": "response",
                        "text": response.message,
                        "slots": response.slots,
                        "stage": response.stage.value,
                    })
                    
                    await channel.send_status("speaking")
                    channel.session.is_speaking = True
                    
                    tts_result = app_state.tts.synthesize(response.message)
                    await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
                    
                    channel.session.is_speaking = False
                    
                    if response.ended:
                        state = agent.get_state()
                        await websocket.send_json({
                            "type": "complete",
                            "confirmation": state.get("confirmation_number"),
                            "slots": response.slots,
                        })
                        logger.info(
                            "appointment_complete",
                            confirmation=state.get("confirmation_number"),
                            session_id=channel.session.session_id[:8],
                        )
                        break
                
                await channel.send_status("listening")
                channel.session.state = SessionState.LISTENING
    
    except WebSocketDisconnect:
        logger.info("healthcare_client_disconnected", session_id=channel.session.session_id[:8])
    except Exception as e:
        logger.error("healthcare_websocket_error", error=str(e), exc_info=True)
    finally:
        agent.end()
        app_state.audit.session_end(
            channel.session.session_id,
            duration_s=channel.session.duration_s,
        )
        await channel.close()


@router.websocket("/ws/text")
async def healthcare_text_websocket(websocket: WebSocket):
    """
    Healthcare TEXT WebSocket endpoint (no audio).
    
    For testing without microphone.
    
    Protocol:
    - Client sends: {"type": "text", "text": "user message"}
    - Server sends: {"type": "response", "text": "...", "slots": {...}}
    - Server sends: {"type": "complete", "confirmation": "..."}
    """
    await websocket.accept()
    
    try:
        healthcare_config = load_healthcare_config("configs/healthcare/clinic.yaml")
    except:
        healthcare_config = load_healthcare_config(None)
    
    agent = HealthcareConversationAgent(
        clinic_name=healthcare_config.clinic.name,
        require_verification=False,
        available_slots=healthcare_config.available_slots,
    )
    
    session_id = "text-" + str(id(websocket))[:8]
    
    logger.info("healthcare_text_session_started", session_id=session_id)
    
    try:
        response = agent.start(session_id=session_id)
        
        await websocket.send_json({
            "type": "response",
            "text": response.message,
            "slots": response.slots,
            "stage": response.stage.value,
        })
        
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            if data.get("type") == "text":
                user_text = data.get("text", "").strip()
                
                if not user_text:
                    continue
                
                logger.info("user_text", text=user_text, session_id=session_id)
                
                response = agent.process(user_text)
                
                logger.info(
                    "agent_response",
                    stage=response.stage.value,
                    slots=list(response.slots.keys()),
                    ended=response.ended,
                )
                
                await websocket.send_json({
                    "type": "response",
                    "text": response.message,
                    "slots": response.slots,
                    "stage": response.stage.value,
                })
                
                if response.ended:
                    state = agent.get_state()
                    await websocket.send_json({
                        "type": "complete",
                        "confirmation": state.get("confirmation_number"),
                        "slots": response.slots,
                    })
                    break
            
            elif data.get("type") == "stop":
                break
    
    except WebSocketDisconnect:
        logger.info("healthcare_text_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("healthcare_text_error", error=str(e))
    finally:
        agent.end()
        try:
            await websocket.close()
        except:
            pass
