"""
WebRTC channel for high-quality audio (48kHz Opus).

This bypasses 8kHz telephony limitations for demos and kiosk applications.
"""
import asyncio
import json
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
import numpy as np

from ..agent import ToolCallingAgent
from ..logging import get_logger

logger = get_logger("channels.webrtc")

# Global relay for media streams
relay = MediaRelay()


@dataclass
class WebRTCSession:
    """Tracks a WebRTC session."""
    session_id: str
    pc: RTCPeerConnection
    agent: ToolCallingAgent
    audio_buffer: list = field(default_factory=list)
    is_speaking: bool = False
    last_speech_time: float = 0


class AudioTrackProcessor:
    """
    Processes incoming audio from WebRTC.
    Accumulates audio and detects speech segments.
    """
    
    def __init__(self, session: WebRTCSession, app_state):
        self.session = session
        self.app_state = app_state
        self.sample_rate = 48000  # WebRTC Opus default
        self.buffer = []
        self.speech_frames = []
        self.is_speech = False
        self.silence_frames = 0
        self.min_silence_frames = 25  # ~500ms at 20ms frames
        self.min_speech_frames = 12   # ~250ms
        
    async def process_frame(self, frame: AudioFrame) -> Optional[np.ndarray]:
        """
        Process an audio frame from WebRTC.
        Returns complete speech segment when detected, None otherwise.
        """
        # Convert to numpy
        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # Mono
        audio = audio.astype(np.float32) / 32768.0  # Normalize
        
        # Resample to 16kHz for VAD/ASR
        if frame.sample_rate != 16000:
            ratio = 16000 / frame.sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]
        
        # VAD
        vad_result = self.app_state.vad.process_chunk(audio)
        
        if vad_result.is_speech:
            self.speech_frames.append(audio)
            self.silence_frames = 0
            self.is_speech = True
        else:
            if self.is_speech:
                self.silence_frames += 1
                self.speech_frames.append(audio)  # Include trailing silence
                
                # Check if speech segment ended
                if self.silence_frames >= self.min_silence_frames:
                    if len(self.speech_frames) >= self.min_speech_frames:
                        # Return complete segment
                        segment = np.concatenate(self.speech_frames)
                        self.speech_frames = []
                        self.is_speech = False
                        self.silence_frames = 0
                        return segment
                    else:
                        # Too short, discard
                        self.speech_frames = []
                        self.is_speech = False
                        self.silence_frames = 0
        
        return None


class WebRTCChannel:
    """
    WebRTC channel manager.
    Handles signaling and audio processing.
    """
    
    def __init__(self, app_state):
        self.app_state = app_state
        self.sessions: Dict[str, WebRTCSession] = {}
        
    async def create_session(self) -> tuple[str, RTCPeerConnection]:
        """Create a new WebRTC session."""
        session_id = str(uuid.uuid4())[:8]
        pc = RTCPeerConnection()
        
        # Initialize agent
        agent = ToolCallingAgent(
            llm=self.app_state.llm,
            clinic_name="Sunrise Medical",
        )
        
        session = WebRTCSession(
            session_id=session_id,
            pc=pc,
            agent=agent,
        )
        self.sessions[session_id] = session
        
        # Track lifecycle
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info("webrtc_state_change", 
                       session=session_id, 
                       state=pc.connectionState)
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                await self.close_session(session_id)
        
        # Handle incoming audio
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                logger.info("webrtc_audio_track_received", session=session_id)
                asyncio.create_task(self._process_audio_track(session, track))
        
        logger.info("webrtc_session_created", session=session_id)
        return session_id, pc
    
    async def _process_audio_track(self, session: WebRTCSession, track: MediaStreamTrack):
        """Process incoming audio track."""
        processor = AudioTrackProcessor(session, self.app_state)
        
        # Send greeting first
        greeting = session.agent.start()
        await self._send_audio_response(session, greeting)
        
        try:
            while True:
                frame = await track.recv()
                segment = await processor.process_frame(frame)
                
                if segment is not None:
                    # Transcribe
                    transcript = self.app_state.asr.transcribe(segment)
                    
                    if transcript.text.strip():
                        user_text = transcript.text.strip()
                        logger.info("webrtc_user_speech", 
                                   session=session.session_id,
                                   text=user_text)
                        
                        # Process through agent
                        response = session.agent.process(user_text)
                        
                        logger.info("webrtc_agent_response",
                                   session=session.session_id,
                                   response=response.text[:80],
                                   state=session.agent.context.state.value)
                        
                        # Send response
                        await self._send_audio_response(session, response.text)
                        
                        if response.ended:
                            if response.booking:
                                logger.info("webrtc_booking_complete",
                                           session=session.session_id,
                                           booking=response.booking)
                            break
                            
        except Exception as e:
            if "Connection" not in str(e):
                logger.error("webrtc_audio_error", error=str(e))
    
    async def _send_audio_response(self, session: WebRTCSession, text: str):
        """Synthesize and queue audio response."""
        try:
            tts_result = self.app_state.tts.synthesize(text)
            # Audio is sent via data channel or separate track
            # For now, we'll use a data channel approach
            logger.info("webrtc_tts_complete", 
                       session=session.session_id,
                       duration_ms=tts_result.duration_ms)
        except Exception as e:
            logger.error("webrtc_tts_error", error=str(e))
    
    async def handle_offer(self, session_id: str, offer_sdp: str) -> str:
        """Handle WebRTC offer and return answer."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await session.pc.setRemoteDescription(offer)
        
        answer = await session.pc.createAnswer()
        await session.pc.setLocalDescription(answer)
        
        return session.pc.localDescription.sdp
    
    async def close_session(self, session_id: str):
        """Close a WebRTC session."""
        session = self.sessions.pop(session_id, None)
        if session:
            await session.pc.close()
            logger.info("webrtc_session_closed", session=session_id)


# Signaling endpoints for HTTP server
async def webrtc_offer_handler(request, app_state):
    """Handle WebRTC offer from client."""
    from aiohttp import web
    
    data = await request.json()
    
    # Get or create channel
    if not hasattr(app_state, 'webrtc_channel'):
        app_state.webrtc_channel = WebRTCChannel(app_state)
    
    channel = app_state.webrtc_channel
    
    if 'session_id' not in data:
        # New session
        session_id, pc = await channel.create_session()
    else:
        session_id = data['session_id']
    
    # Handle offer
    answer_sdp = await channel.handle_offer(session_id, data['sdp'])
    
    return web.json_response({
        'session_id': session_id,
        'sdp': answer_sdp,
        'type': 'answer'
    })
