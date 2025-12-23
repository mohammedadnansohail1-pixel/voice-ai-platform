"""FreeSWITCH channel for SIP/WebRTC calls."""
import asyncio
import base64
import json
import struct
from typing import AsyncIterator, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..core.registry import channel_registry
from ..core.config import Config
from ..core.types import SessionContext, ChannelType
from ..logging import get_logger
from .base import BaseChannel

logger = get_logger("channels.freeswitch")


@channel_registry.register("freeswitch")
class FreeSwitchChannel(BaseChannel):
    """
    FreeSWITCH channel via mod_audio_stream WebSocket.
    
    Supports:
    - L16 (Linear PCM 16-bit) audio at 8kHz or 16kHz
    - Bidirectional streaming
    - Call control events
    
    FreeSWITCH config example:
    <action application="socket" data="$${local_ip}:8000/freeswitch/media-stream async full"/>
    """
    
    def __init__(self, websocket: WebSocket, config: Config) -> None:
        super().__init__(config)
        self.websocket = websocket
        self.is_connected = False
        self.call_uuid: Optional[str] = None
        self.caller_id: Optional[str] = None
        self.destination: Optional[str] = None
        self.sample_rate = 16000  # Default, may be negotiated
        self.session = SessionContext(channel_type=ChannelType.SIP)
        
        # Audio buffer for chunk alignment
        self._audio_buffer = bytearray()
    
    async def accept(self) -> None:
        """Accept WebSocket connection."""
        await self.websocket.accept()
        self.is_connected = True
        logger.info("freeswitch_connected", session_id=self.session.session_id[:8])
    
    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """
        Receive audio from FreeSWITCH.
        
        FreeSWITCH sends L16 (16-bit PCM) audio.
        We convert to float32 for processing.
        """
        try:
            while self.is_connected:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    # Send keepalive
                    await self._send_event("keepalive")
                    continue
                
                if "text" in message:
                    # Control message (JSON)
                    await self._handle_control_message(message["text"])
                
                elif "bytes" in message:
                    # Audio data (L16 PCM)
                    audio_bytes = message["bytes"]
                    
                    # Add to buffer
                    self._audio_buffer.extend(audio_bytes)
                    
                    # Yield chunks of 512 samples (for Silero VAD)
                    chunk_bytes = 512 * 2  # 512 samples * 2 bytes per int16
                    
                    while len(self._audio_buffer) >= chunk_bytes:
                        chunk = bytes(self._audio_buffer[:chunk_bytes])
                        self._audio_buffer = self._audio_buffer[chunk_bytes:]
                        
                        # Convert L16 to float32
                        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32768.0
                        
                        yield audio_float
                        
        except WebSocketDisconnect:
            logger.info("freeswitch_disconnected", session_id=self.session.session_id[:8])
            self.is_connected = False
        except Exception as e:
            logger.error("freeswitch_receive_error", error=str(e))
            self.is_connected = False
    
    async def _handle_control_message(self, message: str) -> None:
        """Handle FreeSWITCH control messages."""
        try:
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connect":
                self.call_uuid = data.get("uuid")
                self.caller_id = data.get("caller_id_number")
                self.destination = data.get("destination_number")
                self.sample_rate = data.get("sample_rate", 16000)
                
                logger.info(
                    "freeswitch_call_connected",
                    uuid=self.call_uuid,
                    caller=self.caller_id,
                    destination=self.destination,
                    sample_rate=self.sample_rate,
                )
                
                # Send acknowledgment
                await self._send_event("connected", {"sample_rate": self.sample_rate})
            
            elif event == "hangup":
                logger.info("freeswitch_call_hangup", uuid=self.call_uuid)
                self.is_connected = False
            
            elif event == "dtmf":
                digit = data.get("digit")
                logger.info("freeswitch_dtmf", digit=digit)
                # Store DTMF for flow processing
                self.session.metadata["last_dtmf"] = digit
            
        except json.JSONDecodeError:
            # Might be ESL event format
            logger.debug("freeswitch_raw_message", message=message[:100])
    
    async def _send_event(self, event: str, data: dict = None) -> None:
        """Send control event to FreeSWITCH."""
        if not self.is_connected:
            return
        
        message = {"event": event}
        if data:
            message.update(data)
        
        await self.websocket.send_json(message)
    
    async def send_audio(self, audio: np.ndarray, sample_rate: int = 24000) -> None:
        """
        Send audio to FreeSWITCH.
        
        Converts to L16 PCM and resamples if needed.
        """
        if not self.is_connected:
            return
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # Convert float32 to L16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Send in chunks to avoid buffer issues
        chunk_size = 320  # 20ms at 16kHz
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            await self.websocket.send_bytes(chunk.tobytes())
    
    async def hangup(self, cause: str = "NORMAL_CLEARING") -> None:
        """Hang up the call."""
        await self._send_event("hangup", {"cause": cause})
        self.is_connected = False
        logger.info("freeswitch_hangup_sent", cause=cause)
    
    async def transfer(self, destination: str) -> None:
        """Transfer call to another destination."""
        await self._send_event("transfer", {"destination": destination})
        logger.info("freeswitch_transfer", destination=destination)
    
    async def play_tone(self, tone: str) -> None:
        """Play a tone (e.g., 'ring', 'busy', 'congestion')."""
        await self._send_event("tone", {"tone": tone})
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("freeswitch_closed", session_id=self.session.session_id[:8])
    
    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if from_rate == to_rate:
            return audio
        
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.arange(new_length) / ratio
        indices_floor = indices.astype(int)
        frac = indices - indices_floor
        
        # Clamp indices
        indices_floor = np.clip(indices_floor, 0, len(audio) - 1)
        indices_ceil = np.clip(indices_floor + 1, 0, len(audio) - 1)
        
        result = audio[indices_floor] * (1 - frac) + audio[indices_ceil] * frac
        return result.astype(np.float32)
