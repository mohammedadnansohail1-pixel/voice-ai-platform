"""WebSocket audio channel."""
import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..core.registry import channel_registry
from ..core.config import Config
from ..core.types import SessionContext, ChannelType
from ..core.exceptions import ChannelDisconnectedError
from ..logging import get_logger
from .base import BaseChannel

logger = get_logger("channels.websocket")


@channel_registry.register("websocket")
class WebSocketChannel(BaseChannel):
    """
    WebSocket channel for real-time audio streaming.
    
    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 pcm>"}
    - Server sends: {"type": "audio", "data": "<base64 pcm>", "sample_rate": 24000}
    - Control: {"type": "start"}, {"type": "stop"}, {"type": "interrupt"}
    """
    
    def __init__(self, websocket: WebSocket, config: Config) -> None:
        super().__init__(config)
        self.websocket = websocket
        self.is_connected = False
        self.session = SessionContext(channel_type=ChannelType.WEBSOCKET)
        
    async def accept(self) -> None:
        """Accept WebSocket connection."""
        await self.websocket.accept()
        self.is_connected = True
        logger.info("websocket_connected", session_id=self.session.session_id[:8])
    
    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """Receive audio chunks from WebSocket client."""
        try:
            while self.is_connected:
                message = await self.websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "audio":
                    # Decode base64 PCM audio
                    audio_bytes = base64.b64decode(data["data"])
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    yield audio
                
                elif data.get("type") == "stop":
                    logger.info("client_requested_stop")
                    break
                
                elif data.get("type") == "interrupt":
                    logger.info("client_requested_interrupt")
                    # Signal barge-in
                    self.session.is_speaking = False
                    
        except WebSocketDisconnect:
            logger.info("websocket_disconnected", session_id=self.session.session_id[:8])
            self.is_connected = False
        except Exception as e:
            logger.error("websocket_receive_error", error=str(e))
            self.is_connected = False
    
    async def send_audio(self, audio: np.ndarray, sample_rate: int = 24000) -> None:
        """Send audio to WebSocket client."""
        if not self.is_connected:
            raise ChannelDisconnectedError("websocket", self.session.session_id)
        
        # Convert to int16 and base64
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
        
        await self.websocket.send_json({
            "type": "audio",
            "data": audio_b64,
            "sample_rate": sample_rate,
        })
    
    async def send_transcript(self, text: str, is_final: bool = True) -> None:
        """Send transcript to client."""
        if not self.is_connected:
            return
        
        await self.websocket.send_json({
            "type": "transcript",
            "text": text,
            "is_final": is_final,
        })
    
    async def send_response(self, text: str) -> None:
        """Send assistant response text to client."""
        if not self.is_connected:
            return
        
        await self.websocket.send_json({
            "type": "response",
            "text": text,
        })
    
    async def send_status(self, status: str) -> None:
        """Send status update to client."""
        if not self.is_connected:
            return
        
        await self.websocket.send_json({
            "type": "status",
            "status": status,
        })
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("websocket_closed", session_id=self.session.session_id[:8])
