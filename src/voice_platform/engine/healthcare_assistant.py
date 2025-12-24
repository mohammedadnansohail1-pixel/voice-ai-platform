"""Healthcare-specific voice assistant."""
import time
from typing import Optional
from datetime import date

import numpy as np

from ..core.config import Config, load_config
from ..core.types import SessionContext, SessionState
from ..core.exceptions import VoicePlatformError
from ..logging import setup_logging, get_logger, AuditLogger
from ..audio import AudioInput, AudioOutput, SpeechAccumulator
from ..vad import SileroVAD
from ..asr import WhisperASR
from ..tts import KokoroTTS
from ..healthcare import (
    HealthcareConfig,
    load_healthcare_config,
    HealthcareConversationAgent,
    AppointmentService,
)

# Optional: for patient lookup
try:
    from verify_core import PatientIdentity
except ImportError:
    PatientIdentity = None

logger = get_logger("engine.healthcare")


class HealthcareVoiceAssistant:
    """
    Healthcare-specific voice assistant.
    
    Pipeline: Mic → VAD → ASR → Healthcare Agent → TTS → Speaker
    
    Features:
    - Medical entity extraction (symptoms, medications)
    - Patient identity verification
    - HIPAA-compliant PHI redaction in logs
    - Human-in-the-loop review for low confidence
    - FHIR-compatible appointment output
    
    Example:
        assistant = HealthcareVoiceAssistant(
            config_path="configs/base.yaml",
            healthcare_config_path="configs/healthcare/clinic.yaml",
        )
        assistant.run()
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        healthcare_config_path: Optional[str] = None,
        patient_data: Optional["PatientIdentity"] = None,
    ) -> None:
        # Load platform config
        self.config = load_config(config_path) if config_path else Config()
        
        # Load healthcare config
        self.healthcare_config = load_healthcare_config(healthcare_config_path)
        
        # Setup logging
        setup_logging(self.config.logging)
        
        # Audit logger with PHI redaction
        self.audit = AuditLogger(
            enabled=self.config.logging.audit_enabled,
            redact_phi=True,  # Always redact PHI in healthcare
            audit_path=self.config.logging.audit_path,
        )
        
        # Components (lazy loaded)
        self._vad: Optional[SileroVAD] = None
        self._asr: Optional[WhisperASR] = None
        self._tts: Optional[KokoroTTS] = None
        
        # Healthcare agent (replaces raw LLM)
        self._agent: Optional[HealthcareConversationAgent] = None
        
        # Audio I/O
        self._audio_in: Optional[AudioInput] = None
        self._audio_out: Optional[AudioOutput] = None
        self._accumulator: Optional[SpeechAccumulator] = None
        
        # Patient data for verification
        self._patient_data = patient_data
        
        # Session
        self.session: Optional[SessionContext] = None
        self.is_running = False
        
        logger.info(
            "healthcare_assistant_initialized",
            clinic=self.healthcare_config.clinic.name,
            verification=self.healthcare_config.verification.enabled,
        )
    
    def _load_components(self) -> None:
        """Load all ML models and components."""
        logger.info("loading_components")
        
        # VAD
        logger.info("loading_vad", backend=self.config.vad.backend)
        self._vad = SileroVAD(self.config.vad)
        
        # ASR
        logger.info("loading_asr", backend=self.config.asr.backend)
        self._asr = WhisperASR(self.config.asr)
        
        # TTS
        logger.info("loading_tts", backend=self.config.tts.backend)
        self._tts = KokoroTTS(self.config.tts)
        
        # Healthcare Agent (instead of raw LLM)
        logger.info("loading_healthcare_agent")
        self._agent = HealthcareConversationAgent(
            clinic_name=self.healthcare_config.clinic.name,
            require_verification=self.healthcare_config.verification.enabled,
            review_confidence_threshold=self.healthcare_config.review.confidence_threshold,
            available_slots=self.healthcare_config.available_slots or None,
        )
        
        # Audio
        self._audio_in = AudioInput(self.config.audio)
        self._audio_out = AudioOutput(sample_rate=24000)
        self._accumulator = SpeechAccumulator(self.config.vad)
        
        logger.info("components_loaded")
    
    def start_session(self) -> SessionContext:
        """Start a new healthcare conversation session."""
        self.session = SessionContext()
        self.session.state = SessionState.LISTENING
        
        self.audit.session_start(
            self.session.session_id,
            channel="local_mic",
        )
        
        # Start healthcare agent
        response = self._agent.start(
            session_id=self.session.session_id,
            patient_data=self._patient_data,
        )
        
        logger.info(
            "healthcare_session_started",
            session_id=self.session.session_id[:8],
            clinic=self.healthcare_config.clinic.name,
        )
        
        return self.session
    
    def end_session(self) -> None:
        """End current session."""
        if self.session:
            self.audit.session_end(
                self.session.session_id,
                duration_s=self.session.duration_s,
            )
            
            # End healthcare agent session
            if self._agent:
                self._agent.end()
            
            # Log final state
            if self._agent:
                state = self._agent.get_state()
                logger.info(
                    "healthcare_session_ended",
                    session_id=self.session.session_id[:8],
                    duration_s=f"{self.session.duration_s:.1f}",
                    final_stage=state.get("stage"),
                    slots_collected=list(state.get("slots", {}).keys()) if state else [],
                )
            
            self.session.state = SessionState.ENDED
            self.session = None
    
    def process_speech(self, audio: np.ndarray) -> Optional[str]:
        """Process speech audio through ASR."""
        if not self._asr:
            raise VoicePlatformError("ASR not loaded", "COMPONENT_NOT_READY")
        
        transcript = self._asr.transcribe(audio)
        
        if transcript.text.strip():
            # Audit with PHI redaction
            self.audit.transcript(
                self.session.session_id if self.session else "unknown",
                transcript.text,
            )
            return transcript.text.strip()
        
        return None
    
    def generate_response(self, user_text: str) -> tuple[str, bool]:
        """
        Generate healthcare agent response to user input.
        
        Returns (response_text, is_ended) tuple.
        """
        if not self._agent or not self.session:
            raise VoicePlatformError("Agent or session not ready", "COMPONENT_NOT_READY")
        
        start_time = time.perf_counter()
        
        # Process through healthcare agent
        response = self._agent.process(user_text)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            "healthcare_response_generated",
            stage=response.stage.value,
            slots=list(response.slots.keys()),
            ended=response.ended,
            latency_ms=f"{latency_ms:.0f}",
        )
        
        return response.message, response.ended
    
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
        """Run the healthcare voice assistant main loop."""
        logger.info("starting_healthcare_assistant")
        
        # Load components
        self._load_components()
        
        # Start session
        self.start_session()
        
        # Start audio capture
        self._audio_in.start()
        self.is_running = True
        
        # Speak greeting
        greeting = self._agent.start(
            session_id=self.session.session_id,
            patient_data=self._patient_data,
        )
        
        clinic_name = self.healthcare_config.clinic.name
        print("\n" + "=" * 60)
        print(f"🏥 {clinic_name} - Healthcare Voice Assistant")
        print("=" * 60)
        print(f"\n🤖 {greeting.message}\n")
        self.speak(greeting.message)
        
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
                    print("🎯 Processing...")
                    text = self.process_speech(segment.audio)
                    
                    if text:
                        print(f"👤 Patient: {text}")
                        
                        # Healthcare Agent
                        response, ended = self.generate_response(text)
                        print(f"🤖 Agent: {response}")
                        
                        # TTS
                        self.speak(response)
                        
                        # Check if conversation ended
                        if ended:
                            print("\n✅ Appointment flow complete!")
                            state = self._agent.get_state()
                            if state.get("confirmation_number"):
                                print(f"📋 Confirmation: {state['confirmation_number']}")
                            self.is_running = False
                    
                    self.session.state = SessionState.LISTENING
        
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for calling!")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the healthcare voice assistant."""
        logger.info("stopping_healthcare_assistant")
        self.is_running = False
        
        if self._audio_in:
            self._audio_in.stop()
        
        if self._audio_out:
            self._audio_out.stop()
        
        self.end_session()
        logger.info("healthcare_assistant_stopped")
