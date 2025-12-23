"""Asterisk AudioSocket channel."""
import asyncio
import struct
from typing import AsyncIterator, Optional, List

import numpy as np

from ..core.config import Config
from ..core.types import SessionContext, ChannelType, LLMMessage
from ..logging import get_logger
from .base import BaseChannel

logger = get_logger("channels.asterisk")


class AsteriskAudioSocket(BaseChannel):
    """Asterisk AudioSocket protocol handler."""
    
    TYPE_UUID = 0x00
    TYPE_SILENCE = 0x01
    TYPE_AUDIO = 0x10
    TYPE_ERROR = 0x11
    TYPE_HANGUP = 0xFF
    
    SAMPLE_RATE = 8000
    TARGET_RATE = 16000
    
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, config: Config):
        super().__init__(config)
        self.reader = reader
        self.writer = writer
        self.is_connected = True
        self.uuid: Optional[str] = None
        self.session = SessionContext(channel_type=ChannelType.SIP)
        self._audio_buffer = bytearray()
        self._sending = False
    
    async def accept(self) -> None:
        logger.info("asterisk_accepting", session_id=self.session.session_id[:8])
        self.is_connected = True
    
    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """Receive audio from Asterisk."""
        try:
            while self.is_connected:
                # Don't process incoming while sending
                if self._sending:
                    await asyncio.sleep(0.02)
                    continue
                
                try:
                    header = await asyncio.wait_for(self.reader.read(3), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                    
                if len(header) < 3:
                    break
                
                msg_type, length = header[0], struct.unpack('>H', header[1:3])[0]
                
                if msg_type == self.TYPE_UUID:
                    uuid_bytes = await self.reader.read(length)
                    self.uuid = uuid_bytes.decode('utf-8').strip('\x00')
                
                elif msg_type == self.TYPE_AUDIO:
                    audio_bytes = await self.reader.read(length)
                    audio_int16 = np.frombuffer(audio_bytes, dtype='<i2')
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    audio_16k = self._upsample(audio_float, self.SAMPLE_RATE, self.TARGET_RATE)
                    
                    self._audio_buffer.extend(audio_16k.astype(np.float32).tobytes())
                    
                    while len(self._audio_buffer) >= 512 * 4:
                        chunk = np.frombuffer(bytes(self._audio_buffer[:512*4]), dtype=np.float32).copy()
                        self._audio_buffer = self._audio_buffer[512*4:]
                        yield chunk
                
                elif msg_type == self.TYPE_SILENCE:
                    if length > 0:
                        await self.reader.read(length)
                
                elif msg_type in (self.TYPE_ERROR, self.TYPE_HANGUP):
                    break
                
                else:
                    if length > 0:
                        await self.reader.read(length)
                        
        except Exception as e:
            logger.error("asterisk_receive_error", error=str(e))
        finally:
            self.is_connected = False
    
    async def send_audio(self, audio: np.ndarray, sample_rate: int = 24000) -> None:
        """Send audio to Asterisk with real-time pacing."""
        if not self.is_connected:
            return
        
        self._sending = True
        
        try:
            if sample_rate != self.SAMPLE_RATE:
                audio = self._upsample(audio, sample_rate, self.SAMPLE_RATE)
            
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype('<i2')
            audio_bytes = audio_int16.tobytes()
            
            # Send in 20ms chunks (160 samples @ 8kHz = 320 bytes) with pacing
            chunk_samples = 160
            chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample
            chunk_duration = chunk_samples / self.SAMPLE_RATE  # 0.02 seconds
            
            for i in range(0, len(audio_bytes), chunk_bytes):
                if not self.is_connected:
                    break
                
                chunk = audio_bytes[i:i + chunk_bytes]
                header = bytes([self.TYPE_AUDIO]) + struct.pack('>H', len(chunk))
                self.writer.write(header + chunk)
                await self.writer.drain()
                
                # Pace at ~real-time (slightly faster to avoid underrun)
                await asyncio.sleep(chunk_duration * 0.9)
            
            logger.info("asterisk_audio_sent", bytes=len(audio_bytes))
        
        finally:
            self._sending = False
    
    async def close(self) -> None:
        if self.is_connected:
            self.is_connected = False
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except:
                pass
    
    def _upsample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate or len(audio) == 0:
            return audio
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        if new_length == 0:
            return np.array([], dtype=np.float32)
        return np.interp(np.linspace(0, len(audio)-1, new_length), np.arange(len(audio)), audio).astype(np.float32)


async def handle_asterisk_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, app_state, config):
    """Handle incoming Asterisk AudioSocket connection."""
    from ..core.types import SessionState
    from ..audio.accumulator import SpeechAccumulator
    
    channel = AsteriskAudioSocket(reader, writer, config)
    await channel.accept()
    
    if not channel.is_connected:
        return
    
    accumulator = SpeechAccumulator(config.vad)
    
    app_state.audit.session_start(channel.session.session_id, channel="asterisk")
    
    messages: List[LLMMessage] = [
        LLMMessage(role="system", content="You are a helpful voice assistant. Keep responses brief - under 2 sentences.")
    ]
    
    try:
        # Send greeting
        greeting = "Hello! How can I help you today?"
        logger.info("asterisk_sending_greeting")
        tts_result = app_state.tts.synthesize(greeting)
        await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
        
        async for audio_chunk in channel.receive_audio():
            vad_result = app_state.vad.process_chunk(audio_chunk)
            segment = accumulator.process(audio_chunk, vad_result)
            
            if segment:
                transcript = app_state.asr.transcribe(segment.audio)
                
                if transcript.text.strip():
                    user_text = transcript.text.strip()
                    logger.info("asterisk_user_said", text=user_text)
                    messages.append(LLMMessage(role="user", content=user_text))
                    
                    logger.info("asterisk_calling_llm")
                    response_text = ""
                    async for token in app_state.llm.generate_stream(messages):
                        response_text += token
                    
                    logger.info("asterisk_llm_response", text=response_text[:80])
                    messages.append(LLMMessage(role="assistant", content=response_text))
                    
                    tts_result = app_state.tts.synthesize(response_text)
                    await channel.send_audio(tts_result.audio_data, tts_result.sample_rate)
                    logger.info("asterisk_response_sent")
    
    except Exception as e:
        logger.error("asterisk_handler_error", error=str(e), exc_info=True)
    finally:
        app_state.audit.session_end(channel.session.session_id, duration_s=channel.session.duration_s)
        await channel.close()
