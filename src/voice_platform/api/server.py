"""FastAPI server for voice platform."""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import Config, load_config
from ..core.types import SessionState
from ..logging import setup_logging, get_logger, AuditLogger
from ..vad import SileroVAD
from ..asr import WhisperASR
from ..llm import OllamaLLM
from ..tts import KokoroTTS
from ..channels.websocket import WebSocketChannel
from ..audio.accumulator import SpeechAccumulator

logger = get_logger("api.server")


class VoicePlatformApp:
    """Voice platform application state."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.vad: Optional[SileroVAD] = None
        self.asr: Optional[WhisperASR] = None
        self.llm: Optional[OllamaLLM] = None
        self.tts: Optional[KokoroTTS] = None
        self.audit: Optional[AuditLogger] = None
        self.is_loaded = False
    
    def load_models(self) -> None:
        """Load all ML models."""
        if self.is_loaded:
            return
            
        logger.info("loading_models")
        
        self.vad = SileroVAD(self.config.vad)
        self.asr = WhisperASR(self.config.asr)
        self.llm = OllamaLLM(self.config.llm)
        self.tts = KokoroTTS(self.config.tts)
        
        self.audit = AuditLogger(
            enabled=self.config.logging.audit_enabled,
            redact_phi=self.config.logging.audit_redact_phi,
            audit_path=self.config.logging.audit_path,
        )
        
        self.is_loaded = True
        logger.info("models_loaded")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application."""
    
    config = load_config(config_path) if config_path else Config()
    setup_logging(config.logging)
    
    # Create app state immediately
    app_state = VoicePlatformApp(config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load models on startup."""
        app_state.load_models()
        logger.info("server_started", tenant=config.tenant.id)
        yield
        logger.info("server_stopped")
    
    app = FastAPI(
        title="Voice AI Platform",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Store state on app
    app.state.platform = app_state
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy" if app_state.is_loaded else "starting",
            "tenant": config.tenant.id,
            "models_loaded": app_state.is_loaded,
        }
    
    @app.get("/health/ready")
    async def ready():
        """Readiness check - returns 503 if models not loaded."""
        if not app_state.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        return {
            "status": "ready",
            "models": {
                "vad": app_state.vad is not None,
                "asr": app_state.asr is not None,
                "llm": app_state.llm is not None,
                "tts": app_state.tts is not None,
            }
        }
    
    @app.websocket("/ws/voice")
    async def voice_websocket(websocket: WebSocket):
        """WebSocket endpoint for voice conversations."""
        if not app_state.is_loaded:
            await websocket.close(code=1013, reason="Models not loaded")
            return
        
        channel = WebSocketChannel(websocket, config)
        await channel.accept()
        
        # Initialize session components
        accumulator = SpeechAccumulator(config.vad)
        
        # Audit
        app_state.audit.session_start(
            channel.session.session_id,
            channel="websocket",
        )
        
        await channel.send_status("ready")
        
        try:
            async for audio_chunk in channel.receive_audio():
                # VAD
                vad_result = app_state.vad.process_chunk(audio_chunk)
                
                # Accumulate speech
                segment = accumulator.process(audio_chunk, vad_result)
                
                if segment:
                    # Speech complete - process
                    await channel.send_status("processing")
                    channel.session.state = SessionState.PROCESSING
                    
                    # ASR
                    transcript = app_state.asr.transcribe(segment.audio)
                    
                    if transcript.text.strip():
                        await channel.send_transcript(transcript.text)
                        
                        # Add to history
                        channel.session.add_message("user", transcript.text)
                        
                        # LLM
                        response = app_state.llm.generate(channel.session.messages)
                        channel.session.add_message("assistant", response.content)
                        
                        await channel.send_response(response.content)
                        
                        # TTS
                        await channel.send_status("speaking")
                        channel.session.state = SessionState.SPEAKING
                        
                        tts_result = app_state.tts.synthesize(response.content)
                        await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
                    
                    await channel.send_status("listening")
                    channel.session.state = SessionState.LISTENING
        
        except WebSocketDisconnect:
            logger.info("client_disconnected", session_id=channel.session.session_id[:8])
        
        finally:
            app_state.audit.session_end(
                channel.session.session_id,
                duration_s=channel.session.duration_s,
            )
            await channel.close()
    
    return app
