"""WebSocket channel for browser-based audio I/O."""
import asyncio
import base64
import json
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from .base import Channel, ChannelEvent, ChannelEventType
from .registry import channel_registry
from ..logging import get_logger

logger = get_logger("channel.websocket")


@channel_registry.register("websocket")
class WebSocketChannel(Channel):
    """
    WebSocket-based audio channel.
    
    Receives audio from browser, sends TTS audio back.
    Audio format: base64-encoded float32 PCM.
    """
    
    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        websocket: Any = None,  # WebSocket connection
    ):
        super().__init__(session_id, sample_rate)
        self.websocket = websocket
        self._receive_task: Optional[asyncio.Task] = None
    
    async def start_async(self) -> None:
        """Start the channel (async version)."""
        if self._is_active:
            return
        
        logger.info("websocket_channel_starting", session_id=self.session_id)
        self._is_active = True
        self._emit(ChannelEvent(type=ChannelEventType.CONNECTED))
        
        # Start receiving audio
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    def start(self) -> None:
        """Start the channel (creates event loop if needed)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.start_async())
            else:
                loop.run_until_complete(self.start_async())
        except RuntimeError:
            asyncio.run(self.start_async())
    
    async def stop_async(self) -> None:
        """Stop the channel (async version)."""
        if not self._is_active:
            return
        
        self._is_active = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        self._emit(ChannelEvent(type=ChannelEventType.DISCONNECTED))
        logger.info("websocket_channel_stopped", session_id=self.session_id)
    
    def stop(self) -> None:
        """Stop the channel."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.stop_async())
            else:
                loop.run_until_complete(self.stop_async())
        except RuntimeError:
            pass
    
    async def play_audio_async(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Send audio to browser for playback."""
        if not self._is_active or not self.websocket:
            return
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio.tobytes()).decode('utf-8')
        
        message = {
            "type": "audio",
            "audio": audio_b64,
            "sample_rate": sample_rate,
        }
        
        try:
            await self.websocket.send_json(message)
            logger.debug("audio_sent", samples=len(audio), sample_rate=sample_rate)
        except Exception as e:
            logger.error("send_audio_error", error=str(e))
    
    def play_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Send audio to browser (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.play_audio_async(audio, sample_rate))
            else:
                loop.run_until_complete(self.play_audio_async(audio, sample_rate))
        except RuntimeError:
            pass
    
    async def interrupt_playback_async(self) -> None:
        """Tell browser to stop playback."""
        if not self._is_active or not self.websocket:
            return
        
        try:
            await self.websocket.send_json({"type": "interrupt"})
        except Exception as e:
            logger.error("interrupt_error", error=str(e))
    
    def interrupt_playback(self) -> None:
        """Interrupt playback (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.interrupt_playback_async())
        except RuntimeError:
            pass
    
    async def send_message(self, text: str) -> None:
        """Send a text message to the browser."""
        if not self._is_active or not self.websocket:
            return
        
        try:
            await self.websocket.send_json({"type": "message", "text": text})
        except Exception as e:
            logger.error("send_message_error", error=str(e))
    
    async def _receive_loop(self) -> None:
        """Receive messages from browser."""
        try:
            while self._is_active and self.websocket:
                try:
                    data = await self.websocket.receive_json()
                    await self._handle_message(data)
                except Exception as e:
                    if self._is_active:
                        logger.error("receive_error", error=str(e))
                        self._emit(ChannelEvent(
                            type=ChannelEventType.ERROR,
                            error=str(e)
                        ))
                    break
        except asyncio.CancelledError:
            pass
    
    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")
        
        if msg_type == "audio":
            # Decode audio
            audio_b64 = data.get("audio", "")
            sample_rate = data.get("sample_rate", self.sample_rate)
            
            audio_bytes = base64.b64decode(audio_b64)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            self._emit(ChannelEvent(
                type=ChannelEventType.AUDIO_RECEIVED,
                audio=audio,
                sample_rate=sample_rate,
            ))
        
        elif msg_type == "speech_start":
            self._emit(ChannelEvent(type=ChannelEventType.SPEECH_START))
        
        elif msg_type == "speech_end":
            self._emit(ChannelEvent(type=ChannelEventType.SPEECH_END))
        
        elif msg_type == "error":
            self._emit(ChannelEvent(
                type=ChannelEventType.ERROR,
                error=data.get("error", "Unknown error")
            ))
