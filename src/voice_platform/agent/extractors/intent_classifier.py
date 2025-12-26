"""
Production LLM-based intent classification and entity extraction.

Fast path: Regex for 80% clear inputs
Slow path: LLM for 20% ambiguous inputs

Used for name, DOB, phone where ASR errors are common.
"""
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from ...llm.ollama import OllamaLLM
from ...core.config import LLMConfig
from ...core.types import LLMMessage
from ...logging import get_logger

logger = get_logger("agent.intent_classifier")


class UserIntent(Enum):
    """Classified user intent."""
    PROVIDING_INFO = "providing_info"      # Giving requested information
    REFUSING = "refusing"                   # Saying no, refusing to provide
    CORRECTING = "correcting"               # Wants to fix something
    CONFIRMING = "confirming"               # Yes, correct, right
    DENYING = "denying"                     # No, that's wrong
    OFF_TOPIC = "off_topic"                 # Unrelated to current question
    UNCLEAR = "unclear"                     # Can't determine


@dataclass
class ClassificationResult:
    """Result from intent classification."""
    intent: UserIntent
    extracted_value: Optional[str] = None
    confidence: float = 0.0
    used_llm: bool = False


class IntentClassifier:
    """
    Hybrid intent classifier with regex fast-path and LLM fallback.
    
    Production pattern:
    1. Try fast regex classification
    2. If ambiguous, use LLM (adds ~200ms)
    3. Cache common patterns
    """
    
    # Fast-path patterns (no LLM needed)
    REFUSAL_PATTERNS = [
        r"^no+\.?$",
        r"^nope\.?$", 
        r"i (don'?t|do not|won'?t|will not|refuse)",
        r"(prefer not|rather not|don'?t want)",
        r"(stop|cancel|end|quit|hang up)",
    ]
    
    CONFIRMATION_PATTERNS = [
        r"^yes+\.?$",
        r"^yeah\.?$",
        r"^yep\.?$",
        r"^(correct|right|exactly|that'?s? (right|correct))\.?$",
        r"^(sure|ok|okay|alright)\.?$",
    ]
    
    DENIAL_PATTERNS = [
        r"^no\.?$",
        r"(that'?s?|it'?s?) (not |in)?correct",
        r"(that'?s?|it'?s?) (wrong|not right)",
        r"(is |was )?(wrong|incorrect|not correct|not right)",
        r"(wrong|incorrect|mistake)",
        r"you (got|have|heard) .*(wrong|incorrect)",
        r"i (didn'?t|did not|never) say",
        r"i said no",
    ]
    
    CORRECTION_PATTERNS = [
        r"(change|correct|fix|update) (my|the|it)",
        r"let me (correct|fix|change)",
        r"go back",
        r"(actually|no),? (my|it'?s?|the)",
        r"my (name|phone|number|date|birthday) is",
    ]
    
    # Name extraction patterns
    NAME_PATTERNS = [
        r"(?:my name is|i'm|i am|this is|call me|it's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$",  # Just a name
    ]
    
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self._llm = llm
        self._llm_initialized = False
        logger.info("intent_classifier_initialized")
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy LLM initialization."""
        if self._llm is None:
            config = LLMConfig(model="llama3.2:latest", temperature=0.1, max_tokens=100)
            self._llm = OllamaLLM(config)
            self._llm_initialized = True
        return self._llm
    
    def classify(
        self, 
        user_input: str, 
        context: str = "general",
        force_llm: bool = False,
    ) -> ClassificationResult:
        """
        Classify user intent with fast-path and LLM fallback.
        
        Args:
            user_input: What the user said
            context: Current context ("name", "dob", "phone", "day", "time", "confirm")
            force_llm: Skip fast-path, always use LLM
            
        Returns:
            ClassificationResult with intent and extracted value
        """
        text = user_input.strip()
        lower = text.lower()
        
        if not force_llm:
            # Fast path: Check clear patterns
            result = self._fast_classify(text, lower, context)
            if result and result.confidence > 0.8:
                logger.debug("fast_path_classification", intent=result.intent.value, confidence=result.confidence)
                return result
        
        # Slow path: Use LLM for ambiguous inputs
        return self._llm_classify(text, context)
    
    def _fast_classify(self, text: str, lower: str, context: str) -> Optional[ClassificationResult]:
        """Fast regex-based classification."""
        
        # Check refusal FIRST (highest priority)
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, lower):
                return ClassificationResult(
                    intent=UserIntent.REFUSING,
                    confidence=0.95,
                    used_llm=False,
                )
        
        # Check denial/correction BEFORE name extraction
        for pattern in self.DENIAL_PATTERNS:
            if re.search(pattern, lower):
                # Check if they're also providing a correction value
                extracted = self._extract_value_from_correction(text, context)
                return ClassificationResult(
                    intent=UserIntent.DENYING,
                    extracted_value=extracted,
                    confidence=0.9,
                    used_llm=False,
                )
        
        # Check confirmation
        for pattern in self.CONFIRMATION_PATTERNS:
            if re.search(pattern, lower):
                return ClassificationResult(
                    intent=UserIntent.CONFIRMING,
                    confidence=0.95,
                    used_llm=False,
                )
        
        # Check correction intent with possible new value
        for pattern in self.CORRECTION_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                extracted = self._extract_value_from_correction(text, context)
                return ClassificationResult(
                    intent=UserIntent.CORRECTING,
                    extracted_value=extracted,
                    confidence=0.9,
                    used_llm=False,
                )
        
        # Context-specific extraction (only if no negative signals)
        if context == "name":
            # Skip name extraction if contains negative words
            if re.search(r"\b(no|not|wrong|incorrect|said no|i said)\b", lower):
                return None  # Trigger LLM
            
            name = self._extract_name_fast(text)
            if name:
                return ClassificationResult(
                    intent=UserIntent.PROVIDING_INFO,
                    extracted_value=name,
                    confidence=0.85,
                    used_llm=False,
                )
        
        # Ambiguous - return None to trigger LLM
        return None
    
    def _extract_name_fast(self, text: str) -> Optional[str]:
        """Fast name extraction with basic validation."""
        # Try patterns
        for pattern in self.NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                # Validation: reasonable name length
                if 2 <= len(name) <= 50 and len(name.split()) <= 4:
                    return name
        
        # Fallback: clean the input and check if it looks like a name
        cleaned = re.sub(r"[.,!?]", "", text).strip()
        words = cleaned.split()
        
        # Filter to likely name words
        name_words = []
        skip_words = {"the", "is", "my", "and", "to", "a", "an", "for", "of", "um", "uh", 
                      "yes", "no", "not", "i", "me", "please", "thank", "thanks", "hi", "hello",
                      "it", "that", "this", "said", "was", "were", "be", "been", "wrong", "correct",
                      "incorrect", "right", "actually", "but", "so", "just", "like", "dont", "didnt"}
        
        for word in words[:3]:
            lower_word = word.lower()
            if lower_word in skip_words:
                continue
            if len(word) >= 2 and word[0].isupper() or word.isalpha():
                name_words.append(word.title())
        
        if name_words:
            name = " ".join(name_words)
            if 2 <= len(name) <= 50:
                return name
        
        return None
    
    def _extract_value_from_correction(self, text: str, context: str) -> Optional[str]:
        """Extract the corrected value from correction statement."""
        lower = text.lower()
        
        if context == "name":
            # "Actually, my name is John Smith"
            match = re.search(r"(?:name is|it'?s?|i'?m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
            if match:
                return match.group(1).title()
        
        return None
    
    def _llm_classify(self, text: str, context: str) -> ClassificationResult:
        """Use LLM for ambiguous classification."""
        
        prompt = f"""Classify this user response in a voice appointment booking system.

Context: Currently asking for {context}
User said: "{text}"

Classify as ONE of:
- PROVIDING_INFO: User is giving the requested information
- REFUSING: User refuses to provide info or wants to stop
- CORRECTING: User wants to fix/correct previously given info
- CONFIRMING: User confirms (yes, correct, right)
- DENYING: User denies (no, that's wrong, incorrect)
- OFF_TOPIC: User is asking/saying something unrelated
- UNCLEAR: Cannot determine intent

Also extract the relevant value if PROVIDING_INFO or CORRECTING.

Respond in this exact format:
INTENT: <intent>
VALUE: <extracted value or NONE>
CONFIDENCE: <0.0-1.0>"""

        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = self.llm.generate(messages, max_tokens=100).content
            
            # Parse response
            intent_match = re.search(r"INTENT:\s*(\w+)", response)
            value_match = re.search(r"VALUE:\s*(.+?)(?:\n|$)", response)
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
            
            intent_str = intent_match.group(1) if intent_match else "UNCLEAR"
            value = value_match.group(1).strip() if value_match else None
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            # Map to enum
            intent_map = {
                "PROVIDING_INFO": UserIntent.PROVIDING_INFO,
                "REFUSING": UserIntent.REFUSING,
                "CORRECTING": UserIntent.CORRECTING,
                "CONFIRMING": UserIntent.CONFIRMING,
                "DENYING": UserIntent.DENYING,
                "OFF_TOPIC": UserIntent.OFF_TOPIC,
                "UNCLEAR": UserIntent.UNCLEAR,
            }
            
            intent = intent_map.get(intent_str.upper(), UserIntent.UNCLEAR)
            
            # Clean value
            if value and value.upper() in ("NONE", "N/A", ""):
                value = None
            
            logger.info(
                "llm_classification",
                text=text[:50],
                intent=intent.value,
                value=value[:20] if value else None,
                confidence=confidence,
            )
            
            return ClassificationResult(
                intent=intent,
                extracted_value=value,
                confidence=confidence,
                used_llm=True,
            )
            
        except Exception as e:
            logger.error("llm_classification_failed", error=str(e))
            return ClassificationResult(
                intent=UserIntent.UNCLEAR,
                confidence=0.0,
                used_llm=True,
            )
