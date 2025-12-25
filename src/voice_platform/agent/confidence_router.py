"""
Confidence-based routing for ASR results.

Three-tier strategy (from CMU Spoken Dialog Research):
- HIGH (>0.85): Implicit confirmation, proceed normally
- MEDIUM (0.5-0.85): Explicit confirmation "Did you say X?"
- LOW (<0.5): Reject and retry "Sorry, I didn't catch that"

This handles the fundamental 8kHz telephony ASR quality issues.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..logging import get_logger

logger = get_logger("agent.confidence_router")


class ConfidenceLevel(Enum):
    """ASR confidence level."""
    HIGH = "high"      # >0.85 - trust it
    MEDIUM = "medium"  # 0.5-0.85 - confirm it
    LOW = "low"        # <0.5 - reject it


@dataclass
class ConfidenceResult:
    """Result of confidence routing."""
    level: ConfidenceLevel
    text: str                          # Original or cleaned text
    confidence: float                  # ASR confidence score
    needs_confirmation: bool = False   # Should we ask "Did you say X?"
    should_reject: bool = False        # Should we ask to repeat?
    confirmation_prompt: Optional[str] = None  # "Did you say 'toothache'?"


class ConfidenceRouter:
    """
    Routes ASR results based on confidence scores.
    
    Thresholds are tuned for Whisper on 8kHz telephony audio.
    """
    
    # Confidence thresholds (Whisper avg_logprob)
    # Note: Whisper uses log probabilities, so values are negative
    # -0.3 is high confidence, -1.0 is low
    HIGH_THRESHOLD = -0.5      # Above this = high confidence
    MEDIUM_THRESHOLD = -0.9    # Above this = medium, below = low
    
    # For normalized 0-1 scores (if ASR provides them)
    HIGH_THRESHOLD_NORM = 0.85
    MEDIUM_THRESHOLD_NORM = 0.50
    
    def __init__(
        self,
        high_threshold: float = HIGH_THRESHOLD_NORM,
        medium_threshold: float = MEDIUM_THRESHOLD_NORM,
        use_logprob: bool = False,
    ):
        """
        Initialize confidence router.
        
        Args:
            high_threshold: Threshold for high confidence
            medium_threshold: Threshold for medium confidence
            use_logprob: If True, expect Whisper log probabilities (negative values)
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.use_logprob = use_logprob
        
        logger.info(
            "confidence_router_initialized",
            high=high_threshold,
            medium=medium_threshold,
            use_logprob=use_logprob,
        )

    def route(
        self,
        text: str,
        confidence: float,
        extracted_slot: Optional[str] = None,
        extracted_value: Optional[str] = None,
    ) -> ConfidenceResult:
        """
        Route ASR result based on confidence.
        
        Args:
            text: Transcribed text
            confidence: ASR confidence score
            extracted_slot: Name of slot extracted (e.g., "visit_reason")
            extracted_value: Value extracted (e.g., "toothache")
            
        Returns:
            ConfidenceResult with routing decision
        """
        # Normalize confidence if using logprob
        if self.use_logprob:
            # Convert log prob to 0-1 scale
            # -0.3 → ~0.95, -0.5 → ~0.85, -1.0 → ~0.50
            norm_confidence = min(1.0, max(0.0, 1.0 + (confidence * 0.5)))
        else:
            norm_confidence = confidence
        
        # Determine level
        if norm_confidence >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif norm_confidence >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        # Build result
        result = ConfidenceResult(
            level=level,
            text=text,
            confidence=norm_confidence,
        )
        
        if level == ConfidenceLevel.LOW:
            result.should_reject = True
            logger.debug(
                "confidence_low_reject",
                confidence=norm_confidence,
                text=text[:30],
            )
            
        elif level == ConfidenceLevel.MEDIUM:
            result.needs_confirmation = True
            
            # Build confirmation prompt
            if extracted_value and extracted_slot:
                if extracted_slot == "visit_reason":
                    result.confirmation_prompt = f"I heard '{extracted_value}'. Is that correct?"
                elif extracted_slot == "preferred_day":
                    result.confirmation_prompt = f"Did you say {extracted_value}?"
                elif extracted_slot == "preferred_time":
                    result.confirmation_prompt = f"Was that {extracted_value}?"
                else:
                    result.confirmation_prompt = f"I heard '{extracted_value}'. Is that right?"
            else:
                result.confirmation_prompt = f"I heard '{text}'. Is that correct?"
                
            logger.debug(
                "confidence_medium_confirm",
                confidence=norm_confidence,
                text=text[:30],
                prompt=result.confirmation_prompt,
            )
        else:
            # High confidence - proceed normally
            logger.debug(
                "confidence_high_proceed",
                confidence=norm_confidence,
                text=text[:30],
            )
        
        return result

    def get_rejection_message(self) -> str:
        """Get a polite rejection message for low confidence."""
        return "Sorry, I didn't catch that. Could you please repeat?"

    def get_retry_message(self, slot_name: str) -> str:
        """Get a retry message for a specific slot."""
        messages = {
            "visit_reason": "Could you tell me again what brings you in today?",
            "preferred_day": "What day would work best for you?",
            "preferred_time": "What time would you prefer?",
        }
        return messages.get(slot_name, "Could you please repeat that?")


class ConfidenceAwareProcessor:
    """
    Wraps slot extraction with confidence handling.
    
    Usage:
        processor = ConfidenceAwareProcessor(extractor, router)
        result = processor.process(text, confidence)
        
        if result.needs_confirmation:
            # Ask: "Did you say X?"
        elif result.should_reject:
            # Ask: "Sorry, didn't catch that"
        else:
            # Proceed normally
    """
    
    def __init__(self, router: Optional[ConfidenceRouter] = None):
        self.router = router or ConfidenceRouter()
        
        # Track confirmation state
        self.pending_confirmation: Optional[ConfidenceResult] = None
        self.confirmation_count: int = 0
        self.max_confirmations: int = 2  # After 2 failed confirmations, proceed anyway
        
    def process(
        self,
        text: str,
        confidence: float,
        extracted_slot: Optional[str] = None,
        extracted_value: Optional[str] = None,
    ) -> ConfidenceResult:
        """
        Process ASR result with confidence routing.
        """
        # Check if this is a response to a pending confirmation
        if self.pending_confirmation:
            return self._handle_confirmation_response(text, confidence)
        
        # Route based on confidence
        result = self.router.route(text, confidence, extracted_slot, extracted_value)
        
        # If medium confidence, store for confirmation
        if result.needs_confirmation:
            self.pending_confirmation = result
            self.confirmation_count = 0
        
        return result

    def _handle_confirmation_response(
        self,
        text: str,
        confidence: float,
    ) -> ConfidenceResult:
        """Handle response to a confirmation request."""
        text_lower = text.lower().strip()
        
        # Check for yes/no
        yes_words = ["yes", "yeah", "yep", "correct", "right", "that's right"]
        no_words = ["no", "nope", "wrong", "incorrect"]
        
        is_yes = any(w in text_lower for w in yes_words)
        is_no = any(w in text_lower for w in no_words)
        
        if is_yes:
            # Confirmed - return original with high confidence
            result = self.pending_confirmation
            result.level = ConfidenceLevel.HIGH
            result.needs_confirmation = False
            self.pending_confirmation = None
            
            logger.info("confirmation_accepted", text=result.text[:30])
            return result
            
        elif is_no:
            # Rejected - ask again
            self.confirmation_count += 1
            self.pending_confirmation = None
            
            logger.info("confirmation_rejected", count=self.confirmation_count)
            
            return ConfidenceResult(
                level=ConfidenceLevel.LOW,
                text=text,
                confidence=confidence,
                should_reject=True,
            )
        else:
            # Unclear response
            self.confirmation_count += 1
            
            if self.confirmation_count >= self.max_confirmations:
                # Too many attempts, proceed with original
                result = self.pending_confirmation
                result.needs_confirmation = False
                self.pending_confirmation = None
                
                logger.warning(
                    "confirmation_max_attempts",
                    proceeding_with=result.text[:30],
                )
                return result
            
            # Ask again
            return self.pending_confirmation
    
    def clear_pending(self):
        """Clear any pending confirmation."""
        self.pending_confirmation = None
        self.confirmation_count = 0
