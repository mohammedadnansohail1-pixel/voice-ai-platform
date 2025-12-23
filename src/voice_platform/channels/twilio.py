"""Twilio telephony channel for voice calls."""
import base64
import json
from typing import AsyncIterator, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..core.registry import channel_registry
from ..core.config import Config
from ..core.types import SessionContext, ChannelType
from ..logging import get_logger
from .base import BaseChannel

logger = get_logger("channels.twilio")


@channel_registry.register("twilio")
class TwilioChannel(BaseChannel):
    """
    Twilio Media Streams channel.
    
    Handles bidirectional audio streaming via WebSocket.
    Audio format: mulaw 8kHz (Twilio standard)
    """
    
    TWILIO_SAMPLE_RATE = 8000
    MULAW_CHANNELS = 1
    
    def __init__(self, websocket: WebSocket, config: Config) -> None:
        super().__init__(config)
        self.websocket = websocket
        self.is_connected = False
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.session = SessionContext(channel_type=ChannelType.TWILIO)
        
        # Audio buffer for resampling
        self._audio_buffer = bytearray()
    
    async def accept(self) -> None:
        """Accept WebSocket connection."""
        await self.websocket.accept()
        self.is_connected = True
        logger.info("twilio_websocket_connected", session_id=self.session.session_id[:8])
    
    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """
        Receive audio from Twilio Media Streams.
        
        Twilio sends mulaw 8kHz audio. We convert to PCM float32 16kHz.
        """
        try:
            while self.is_connected:
                message = await self.websocket.receive_text()
                data = json.loads(message)
                event = data.get("event")
                
                if event == "connected":
                    logger.info("twilio_stream_connected")
                
                elif event == "start":
                    self.stream_sid = data.get("streamSid")
                    self.call_sid = data["start"].get("callSid")
                    logger.info(
                        "twilio_stream_started",
                        stream_sid=self.stream_sid,
                        call_sid=self.call_sid,
                    )
                
                elif event == "media":
                    # Decode mulaw audio
                    payload = data["media"]["payload"]
                    audio_bytes = base64.b64decode(payload)
                    
                    # Convert mulaw to PCM float32
                    audio_pcm = self._mulaw_to_pcm(audio_bytes)
                    
                    # Resample 8kHz -> 16kHz
                    audio_16k = self._resample_8k_to_16k(audio_pcm)
                    
                    # Yield chunks of 512 samples (for Silero VAD)
                    self._audio_buffer.extend(audio_16k.tobytes())
                    
                    while len(self._audio_buffer) >= 512 * 4:  # 512 float32 samples
                        chunk_bytes = bytes(self._audio_buffer[:512 * 4])
                        self._audio_buffer = self._audio_buffer[512 * 4:]
                        yield np.frombuffer(chunk_bytes, dtype=np.float32)
                
                elif event == "stop":
                    logger.info("twilio_stream_stopped", stream_sid=self.stream_sid)
                    break
                    
        except WebSocketDisconnect:
            logger.info("twilio_websocket_disconnected", session_id=self.session.session_id[:8])
            self.is_connected = False
        except Exception as e:
            logger.error("twilio_receive_error", error=str(e))
            self.is_connected = False
    
    async def send_audio(self, audio: np.ndarray, sample_rate: int = 24000) -> None:
        """
        Send audio to Twilio.
        
        Converts from PCM to mulaw 8kHz.
        """
        if not self.is_connected or not self.stream_sid:
            return
        
        # Resample to 8kHz if needed
        if sample_rate != self.TWILIO_SAMPLE_RATE:
            audio = self._resample_to_8k(audio, sample_rate)
        
        # Convert to mulaw
        mulaw_bytes = self._pcm_to_mulaw(audio)
        
        # Send in chunks (Twilio expects smaller payloads)
        chunk_size = 640  # ~80ms at 8kHz
        for i in range(0, len(mulaw_bytes), chunk_size):
            chunk = mulaw_bytes[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": payload
                }
            }
            
            await self.websocket.send_json(message)
    
    async def send_mark(self, name: str) -> None:
        """Send a mark event to track audio playback position."""
        if not self.is_connected or not self.stream_sid:
            return
        
        message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": name}
        }
        await self.websocket.send_json(message)
    
    async def clear_audio(self) -> None:
        """Clear pending audio (for barge-in)."""
        if not self.is_connected or not self.stream_sid:
            return
        
        message = {
            "event": "clear",
            "streamSid": self.stream_sid
        }
        await self.websocket.send_json(message)
        logger.debug("twilio_audio_cleared")
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("twilio_websocket_closed", session_id=self.session.session_id[:8])
    
    def _mulaw_to_pcm(self, mulaw_bytes: bytes) -> np.ndarray:
        """Convert mulaw to PCM float32."""
        # Mulaw decoding table
        MULAW_DECODE = np.array([
            -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
            -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
            -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
            -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
            -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
            -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
            -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
            -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
            -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
            -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
            -876, -844, -812, -780, -748, -716, -684, -652,
            -620, -588, -556, -524, -492, -460, -428, -396,
            -372, -356, -340, -324, -308, -292, -276, -260,
            -244, -228, -212, -196, -180, -164, -148, -132,
            -120, -112, -104, -96, -88, -80, -72, -64,
            -56, -48, -40, -32, -24, -16, -8, 0,
            32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
            23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
            15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
            11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
            7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
            5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
            3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
            2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
            1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
            1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
            876, 844, 812, 780, 748, 716, 684, 652,
            620, 588, 556, 524, 492, 460, 428, 396,
            372, 356, 340, 324, 308, 292, 276, 260,
            244, 228, 212, 196, 180, 164, 148, 132,
            120, 112, 104, 96, 88, 80, 72, 64,
            56, 48, 40, 32, 24, 16, 8, 0
        ], dtype=np.int16)
        
        indices = np.frombuffer(mulaw_bytes, dtype=np.uint8)
        pcm_int16 = MULAW_DECODE[indices]
        return pcm_int16.astype(np.float32) / 32768.0
    
    def _pcm_to_mulaw(self, audio: np.ndarray) -> bytes:
        """Convert PCM float32 to mulaw."""
        # Scale to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Mulaw encoding
        MULAW_MAX = 0x1FFF
        MULAW_BIAS = 33
        
        sign = (audio_int16 >> 8) & 0x80
        audio_int16 = np.abs(audio_int16)
        audio_int16 = np.clip(audio_int16, 0, MULAW_MAX)
        audio_int16 = audio_int16 + MULAW_BIAS
        
        # Find segment
        exponent = np.floor(np.log2(audio_int16)).astype(np.int32)
        exponent = np.clip(exponent - 6, 0, 7)
        
        mantissa = (audio_int16 >> (exponent + 3)) & 0x0F
        mulaw = ~(sign | (exponent << 4) | mantissa)
        
        return mulaw.astype(np.uint8).tobytes()
    
    def _resample_8k_to_16k(self, audio: np.ndarray) -> np.ndarray:
        """Simple linear interpolation upsample 8kHz -> 16kHz."""
        # Double the samples via linear interpolation
        n = len(audio)
        indices = np.arange(0, n - 1, 0.5)
        indices_floor = indices.astype(int)
        frac = indices - indices_floor
        
        result = audio[indices_floor] * (1 - frac) + audio[np.minimum(indices_floor + 1, n - 1)] * frac
        return result.astype(np.float32)
    
    def _resample_to_8k(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Downsample to 8kHz."""
        ratio = source_rate / self.TWILIO_SAMPLE_RATE
        new_length = int(len(audio) / ratio)
        indices = (np.arange(new_length) * ratio).astype(int)
        return audio[indices]
