"""Main voice assistant orchestrator."""
import asyncio
import time
from typing import Optional

import numpy as np

from ..core.config import Config, load_config
from ..core.types import SessionContext, SessionState, LLMMessage
from ..core.exceptions import VoicePlatformError
from ..logging import setup_logging, get_logger, AuditLogger
from ..audio import AudioInput, AudioOutput, SpeechAccumulator
from ..vad import SileroVAD
from ..asr import WhisperASR
from ..llm import OllamaLLM
from ..tts import KokoroTTS

logger = get_logger("engine.assistant")


class VoiceAssistant:
    """
    Main voice assistant orchestrator.
    
    Pipeline: Mic → VAD → ASR → LLM → TTS → Speaker
    Supports barge-in interruption.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        # Load config
        self.config = load_config(config_path) if config_path else Config()
        
        # Setup logging
        setup_logging(self.config.logging)
        
        # Audit logger
        self.audit = AuditLogger(
            enabled=self.config.logging.audit_enabled,
            redact_phi=self.config.logging.audit_redact_phi,
            audit_path=self.config.logging.audit_path,
        )
        
        # Components (lazy loaded)
        self._vad: Optional[SileroVAD] = None
        self._asr: Optional[WhisperASR] = None
        self._llm: Optional[OllamaLLM] = None
        self._tts: Optional[KokoroTTS] = None
        
        # Audio I/O
        self._audio_in: Optional[AudioInput] = None
        self._audio_out: Optional[AudioOutput] = None
        self._accumulator: Optional[SpeechAccumulator] = None
        
        # Session
        self.session: Optional[SessionContext] = None
        self.is_running = False
        
        logger.info("voice_assistant_initialized", tenant=self.config.tenant.id)
    
    def _load_components(self) -> None:
        """Load all ML models and components."""
        logger.info("loading_components")
        
        # VAD
        logger.info("loading_vad", backend=self.config.vad.backend)
        self._vad = SileroVAD(self.config.vad)
        
        # ASR
        logger.info("loading_asr", backend=self.config.asr.backend)
        self._asr = WhisperASR(self.config.asr)
        
        # LLM
        logger.info("loading_llm", provider=self.config.llm.provider)
        self._llm = OllamaLLM(self.config.llm)
        
        # TTS
        logger.info("loading_tts", backend=self.config.tts.backend)
        self._tts = KokoroTTS(self.config.tts)
        
        # Audio
        self._audio_in = AudioInput(self.config.audio)
        self._audio_out = AudioOutput(sample_rate=24000)
        self._accumulator = SpeechAccumulator(self.config.vad)
        
        logger.info("components_loaded")
    
    def start_session(self) -> SessionContext:
        """Start a new conversation session."""
        self.session = SessionContext()
        self.session.state = SessionState.LISTENING
        
        # Add system prompt to history
        if self.config.llm.system_prompt:
            self.session.add_message("system", self.config.llm.system_prompt)
        
        self.audit.session_start(self.session.session_id, channel="local_mic")
        
        logger.info("session_started", session_id=self.session.session_id[:8])
        return self.session
    
    def end_session(self) -> None:
        """End current session."""
        if self.session:
            self.audit.session_end(
                self.session.session_id,
                duration_s=self.session.duration_s,
            )
            logger.info(
                "session_ended",
                session_id=self.session.session_id[:8],
                duration_s=f"{self.session.duration_s:.1f}",
            )
            self.session.state = SessionState.ENDED
            self.session = None
    
    def process_speech(self, audio: np.ndarray) -> Optional[str]:
        """
        Process speech audio through ASR.
        
        Returns transcribed text or None.
        """
        if not self._asr:
            raise VoicePlatformError("ASR not loaded", "COMPONENT_NOT_READY")
        
        transcript = self._asr.transcribe(audio)
        
        if transcript.text.strip():
            self.audit.transcript(
                self.session.session_id if self.session else "unknown",
                transcript.text,
            )
            return transcript.text.strip()
        
        return None
    
    def generate_response(self, user_text: str) -> str:
        """Generate LLM response to user input."""
        if not self._llm or not self.session:
            raise VoicePlatformError("LLM or session not ready", "COMPONENT_NOT_READY")
        
        # Add user message to history
        self.session.add_message("user", user_text)
        
        # Generate response
        response = self._llm.generate(self.session.messages)
        
        # Add assistant response to history
        self.session.add_message("assistant", response.content)
        
        logger.debug(
            "llm_response_generated",
            tokens=response.tokens_used,
            latency_ms=f"{response.latency_ms:.0f}",
        )
        
        return response.content
    
    def speak(self, text: str) -> None:
        """Synthesize and play speech."""
        if not self._tts or not self._audio_out:
            raise VoicePlatformError("TTS not ready", "COMPONENT_NOT_READY")
        
        if self.session:
            self.session.state = SessionState.SPEAKING
            self.session.is_speaking = True
        
        result = self._tts.synthesize(text)
        self._audio_out.play(result.audio_data, result.sample_rate)
        
        if self.session:
            self.session.state = SessionState.LISTENING
            self.session.is_speaking = False
    
    def interrupt(self) -> None:
        """Interrupt current speech (barge-in)."""
        if self._audio_out:
            self._audio_out.stop()
            logger.debug("speech_interrupted")
    
    def run(self) -> None:
        """Run the voice assistant main loop."""
        logger.info("starting_voice_assistant")
        
        # Load components
        self._load_components()
        
        # Start session
        self.start_session()
        
        # Start audio capture
        self._audio_in.start()
        self.is_running = True
        
        print("\n" + "=" * 50)
        print("🎤 Voice Assistant Ready - Speak now!")
        print("   Press Ctrl+C to exit")
        print("=" * 50 + "\n")
        
        try:
            while self.is_running:
                # Get audio chunk
                chunk = self._audio_in.get_chunk(timeout=0.1)
                if chunk is None:
                    continue
                
                # Run VAD
                vad_result = self._vad.process_chunk(chunk)
                
                # Check for barge-in during playback
                if self.session and self.session.is_speaking:
                    if vad_result.is_speech and self.config.barge_in.enabled:
                        if vad_result.confidence > self.config.barge_in.energy_threshold:
                            self.interrupt()
                            self._accumulator.reset()
                            continue
                
                # Accumulate speech
                segment = self._accumulator.process(chunk, vad_result)
                
                if segment:
                    # Process complete speech segment
                    self.session.state = SessionState.PROCESSING
                    
                    # ASR
                    print("🎯 Processing speech...")
                    text = self.process_speech(segment.audio)
                    
                    if text:
                        print(f"👤 You: {text}")
                        
                        # LLM
                        response = self.generate_response(text)
                        print(f"🤖 Assistant: {response}")
                        
                        # TTS
                        self.speak(response)
                    
                    self.session.state = SessionState.LISTENING
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the voice assistant."""
        logger.info("stopping_voice_assistant")
        self.is_running = False
        
        if self._audio_in:
            self._audio_in.stop()
        
        if self._audio_out:
            self._audio_out.stop()
        
        self.end_session()
        logger.info("voice_assistant_stopped")
