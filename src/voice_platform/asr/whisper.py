"""Faster-Whisper ASR with contextual biasing and LLM correction."""
import time
from typing import Optional, List
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from ..core.registry import asr_registry
from ..core.config import ASRConfig
from ..core.types import Transcript, TranscriptSegment
from ..logging import get_logger
from .base import BaseASR

logger = get_logger("asr.whisper")


@dataclass
class ASRContext:
    """Context for biasing ASR transcription."""
    domain: str = "healthcare"
    expected_entities: List[str] = None
    conversation_state: str = ""
    previous_turns: List[str] = None
    
    def __post_init__(self):
        if self.expected_entities is None:
            self.expected_entities = []
        if self.previous_turns is None:
            self.previous_turns = []


# Domain-specific prompts for contextual biasing
DOMAIN_PROMPTS = {
    "healthcare": """Medical clinic appointment scheduling phone call.
Patient mentions: Monday, Tuesday, Wednesday, Thursday, Friday.
Times: 9 AM, 10 AM, 11 AM, 2 PM, 3 PM, 4 PM.
Symptoms: toothache, tooth pain, back pain, headache, checkup, cleaning, cavity.
Confirmations: Yes, Yeah, No, Correct, That's right.""",

    "healthcare_day": """Healthcare appointment scheduling. Patient selecting a day.
Days of the week: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
Common responses: Monday works, Tuesday is good, How about Thursday, Friday please, Let's do Wednesday.
Similar sounding: Tuesday and Thursday sound similar.""",

    "healthcare_time": """Healthcare appointment scheduling. Patient selecting a time.
Times: 9 AM, 10 AM, 11 AM, 12 PM, 1 PM, 2 PM, 3 PM, 4 PM, 5 PM.
Phrases: 2 PM works, How about 10 AM, 3 o'clock, morning, afternoon.""",

    "healthcare_confirm": """Healthcare appointment confirmation.
Confirmations: Yes, Yeah, Yep, Sure, OK, Correct, That's right, Book it, Sounds good.
Rejections: No, Nope, Cancel, Wrong, Change it, Different time.""",
}

# State-to-prompt mapping
STATE_PROMPTS = {
    "collecting_reason": "healthcare",
    "collecting_day": "healthcare_day",
    "confirming_day": "healthcare_day", 
    "collecting_time": "healthcare_time",
    "confirming": "healthcare_confirm",
}


@asr_registry.register("whisper")
class WhisperASR(BaseASR):
    """
    Faster-Whisper ASR with contextual biasing.
    
    Features:
    - Domain-specific prompting to bias toward expected vocabulary
    - State-aware prompts that adapt to conversation flow
    - LLM post-correction for ambiguous transcriptions
    """

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        if config is None:
            config = ASRConfig()
        super().__init__(config)

        self.model: Optional[WhisperModel] = None
        self._load_model()
        
        # LLM for post-correction (lazy loaded)
        self._llm = None
        
        # Current context for biasing
        self.context = ASRContext()

    def _load_model(self) -> None:
        """Load Whisper model."""
        logger.info(
            "loading_whisper",
            model=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        start = time.perf_counter()

        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        load_time = time.perf_counter() - start
        logger.info("whisper_loaded", load_time_s=f"{load_time:.2f}")

    def set_context(
        self,
        state: str = "",
        previous_turns: List[str] = None,
        expected_entities: List[str] = None,
    ) -> None:
        """
        Set context for biasing transcription.
        
        Args:
            state: Current conversation state (e.g., "collecting_day")
            previous_turns: Recent conversation turns for context
            expected_entities: Specific words/phrases expected
        """
        self.context.conversation_state = state
        if previous_turns:
            self.context.previous_turns = previous_turns[-4:]  # Keep last 4 turns
        if expected_entities:
            self.context.expected_entities = expected_entities
            
        logger.debug("asr_context_updated", state=state, entities=expected_entities)

    def _build_prompt(self) -> str:
        """Build context-aware prompt for Whisper."""
        # Get base domain prompt
        state = self.context.conversation_state
        prompt_key = STATE_PROMPTS.get(state, "healthcare")
        base_prompt = DOMAIN_PROMPTS.get(prompt_key, DOMAIN_PROMPTS["healthcare"])
        
        # Add expected entities if any
        if self.context.expected_entities:
            entities = ", ".join(self.context.expected_entities)
            base_prompt += f"\nExpected words: {entities}."
        
        # Add recent conversation context
        if self.context.previous_turns:
            recent = " ".join(self.context.previous_turns[-2:])
            if len(recent) < 200:  # Keep prompt reasonable size
                base_prompt += f"\nRecent conversation: {recent}"
        
        return base_prompt

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Transcribe audio using Whisper with contextual biasing.
        """
        # Minimum audio duration
        min_duration = 0.8
        audio_duration = audio.shape[0] / sample_rate
        if audio_duration < min_duration:
            logger.debug("audio_too_short", duration=audio_duration)
            return Transcript(text="", segments=[], duration=audio_duration)

        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        start = time.perf_counter()

        # Build context-aware prompt
        prompt = self._build_prompt()

        segments, info = self.model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=5,
            best_of=5,
            patience=1.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=100,
                min_speech_duration_ms=250,
            ),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )

        # Collect segments
        transcript_segments = []
        full_text_parts = []

        for seg in segments:
            transcript_segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=seg.avg_logprob,
                    language=info.language,
                )
            )
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        # Filter low-confidence transcripts
        if transcript_segments:
            avg_confidence = sum(s.confidence for s in transcript_segments) / len(transcript_segments)
            if avg_confidence < -0.8 and info.language_probability < 0.7:
                logger.warning("low_confidence_transcript", 
                             confidence=avg_confidence, 
                             lang_prob=info.language_probability, 
                             text=full_text[:50])
                return Transcript(text="", segments=[], duration=audio_duration, language=info.language)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "transcription_complete",
            text_length=len(full_text),
            segments=len(transcript_segments),
            language=info.language,
            latency_ms=f"{latency_ms:.1f}",
            prompt_state=self.context.conversation_state,
        )

        return Transcript(
            text=full_text,
            segments=transcript_segments,
            language=info.language,
            duration=audio_duration,
        )

    def transcribe_stream(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> Transcript:
        """Transcribe accumulated audio chunks."""
        if not audio_chunks:
            return Transcript(text="", segments=[], duration=0.0)

        audio = np.concatenate(audio_chunks)
        return self.transcribe(audio, sample_rate)
