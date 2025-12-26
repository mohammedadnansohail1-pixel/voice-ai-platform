"""FastAPI server for voice platform with streaming and telephony."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..core.config import Config, load_config
from ..core.types import SessionState
from ..logging import setup_logging, get_logger, AuditLogger
from ..vad import SileroVAD
from ..asr import WhisperASR
from ..llm import OllamaLLM
from ..tts import KokoroTTS
from ..channels.websocket import WebSocketChannel
from ..channels.twilio import TwilioChannel
from ..channels.freeswitch import FreeSwitchChannel
from ..audio.accumulator import SpeechAccumulator
from ..llm.streaming import StreamingPipeline
from .routes.telephony import router as telephony_router
from .routes.healthcare import router as healthcare_router

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
        self.flow_engine = None
        self.is_loaded = False

    def load_models(self) -> None:
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

        self.flow_engine = None
        self.is_loaded = True
        logger.info("models_loaded")

    def create_streaming_pipeline(self) -> StreamingPipeline:
        return StreamingPipeline(self.llm, self.tts, self.config)


def create_app(config_path: Optional[str] = None) -> FastAPI:
    config = load_config(config_path) if config_path else Config()
    setup_logging(config.logging)

    app_state = VoicePlatformApp(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_state.load_models()
        logger.info("server_started", tenant=config.tenant.id)
        yield
        logger.info("server_stopped")

    app = FastAPI(
        title="Voice AI Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.platform = app_state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(telephony_router)
    app.include_router(healthcare_router)

    static_path = Path(__file__).parent.parent.parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/")
    async def index():
        index_path = static_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Voice AI Platform API", "docs": "/docs"}

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if app_state.is_loaded else "starting",
            "tenant": config.tenant.id,
            "models_loaded": app_state.is_loaded,
        }

    @app.get("/health/ready")
    async def ready():
        if not app_state.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        return {
            "status": "ready",
            "models": {
                "vad": app_state.vad is not None,
                "asr": app_state.asr is not None,
                "llm": app_state.llm is not None,
                "tts": app_state.tts is not None,
            },
            "flows": list(app_state.flow_engine.flows.keys()) if app_state.flow_engine else [],
            "channels": ["websocket", "twilio", "freeswitch", "healthcare"],
        }

    async def handle_voice_session(channel, app_state, config):
        """Common voice session handler for all channels."""
        accumulator = SpeechAccumulator(config.vad)
        pipeline = app_state.create_streaming_pipeline()

        flow_result = None
        if app_state.flow_engine and "greeting" in app_state.flow_engine.flows:
            flow_result = app_state.flow_engine.start_flow(channel.session.session_id, "greeting")

        app_state.audit.session_start(channel.session.session_id, channel=channel.__class__.__name__)

        try:
            # Send initial greeting
            if flow_result and flow_result.say:
                tts_result = app_state.tts.synthesize(flow_result.say)
                await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)

            async for audio_chunk in channel.receive_audio():
                # Barge-in detection
                if channel.session.is_speaking:
                    vad_result = app_state.vad.process_chunk(audio_chunk)
                    if vad_result.is_speech and config.barge_in.enabled:
                        pipeline.interrupt()
                        if hasattr(channel, 'clear_audio'):
                            await channel.clear_audio()
                        channel.session.is_speaking = False
                        accumulator.reset()
                        continue

                vad_result = app_state.vad.process_chunk(audio_chunk)
                segment = accumulator.process(audio_chunk, vad_result)

                if segment:
                    channel.session.state = SessionState.PROCESSING
                    transcript = app_state.asr.transcribe(segment.audio)

                    if transcript.text.strip():
                        user_text = transcript.text.strip()
                        channel.session.add_message("user", user_text)

                        response_text = None
                        if app_state.flow_engine:
                            ctx = app_state.flow_engine.get_context(channel.session.session_id)
                            if ctx:
                                flow_result = app_state.flow_engine.process_input(
                                    channel.session.session_id, user_text
                                )
                                response_text = flow_result.say

                                if flow_result.end_flow:
                                    if response_text:
                                        tts_result = app_state.tts.synthesize(response_text)
                                        await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
                                    break

                        if not response_text:
                            channel.session.state = SessionState.SPEAKING
                            channel.session.is_speaking = True

                            async def on_audio(audio: np.ndarray, sample_rate: int):
                                if not pipeline.is_interrupted:
                                    await channel.send_audio(audio, sample_rate)

                            response_text, _ = await pipeline.generate_streaming(
                                channel.session.messages, on_audio=on_audio
                            )
                            channel.session.is_speaking = False
                        else:
                            tts_result = app_state.tts.synthesize(response_text)
                            await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)

                        channel.session.add_message("assistant", response_text)

                    channel.session.state = SessionState.LISTENING

        except WebSocketDisconnect:
            logger.info("channel_disconnected", session_id=channel.session.session_id[:8])
        finally:
            if app_state.flow_engine:
                app_state.flow_engine.end_flow(channel.session.session_id)
            app_state.audit.session_end(channel.session.session_id, duration_s=channel.session.duration_s)
            await channel.close()

    @app.websocket("/ws/voice")
    async def voice_websocket(websocket: WebSocket):
        if not app_state.is_loaded:
            await websocket.close(code=1013, reason="Models not loaded")
            return

        channel = WebSocketChannel(websocket, config)
        await channel.accept()
        await channel.send_status("ready")

        accumulator = SpeechAccumulator(config.vad)
        pipeline = app_state.create_streaming_pipeline()

        app_state.audit.session_start(channel.session.session_id, channel="websocket")
        await channel.send_status("ready")

        try:
            async for audio_chunk in channel.receive_audio():
                if channel.session.is_speaking:
                    vad_result = app_state.vad.process_chunk(audio_chunk)
                    if vad_result.is_speech and config.barge_in.enabled:
                        pipeline.interrupt()
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
                        await channel.send_transcript(transcript.text)
                        channel.session.add_message("user", transcript.text)

                        await channel.send_status("speaking")
                        channel.session.state = SessionState.SPEAKING
                        channel.session.is_speaking = True

                        async def on_audio(audio: np.ndarray, sample_rate: int):
                            if not pipeline.is_interrupted:
                                await channel.send_audio(audio, sample_rate)

                        full_response, _ = await pipeline.generate_streaming(
                            channel.session.messages, on_audio=on_audio
                        )

                        channel.session.add_message("assistant", full_response)
                        await channel.send_response(full_response)
                        channel.session.is_speaking = False

                    await channel.send_status("listening")
                    channel.session.state = SessionState.LISTENING

        except WebSocketDisconnect:
            logger.info("client_disconnected", session_id=channel.session.session_id[:8])
        finally:
            app_state.audit.session_end(channel.session.session_id, duration_s=channel.session.duration_s)
            await channel.close()

    @app.websocket("/telephony/media-stream")
    async def twilio_media_stream(websocket: WebSocket):
        """Twilio Media Streams endpoint."""
        if not app_state.is_loaded:
            await websocket.close(code=1013, reason="Models not loaded")
            return

        channel = TwilioChannel(websocket, config)
        await channel.accept()
        await handle_voice_session(channel, app_state, config)

    @app.websocket("/freeswitch/media-stream")
    async def freeswitch_media_stream(websocket: WebSocket):
        """FreeSWITCH audio stream endpoint."""
        if not app_state.is_loaded:
            await websocket.close(code=1013, reason="Models not loaded")
            return

        channel = FreeSwitchChannel(websocket, config)
        await channel.accept()
        await handle_voice_session(channel, app_state, config)

    return app
