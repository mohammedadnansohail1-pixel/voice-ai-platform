"""
LLM-based ASR correction for ambiguous transcriptions.
"""
from typing import Optional, List, Dict
from dataclasses import dataclass
import re

from ..logging import get_logger

logger = get_logger("asr.correction")


@dataclass
class CorrectionResult:
    """Result of ASR correction."""
    original: str
    corrected: str
    was_corrected: bool
    confidence: float
    corrections_made: List[str]


# Common ASR mistakes for healthcare domain
PHONETIC_CORRECTIONS = {
    # Days - Tuesday/Thursday confusion
    "choose day": "Tuesday",
    "shoes day": "Tuesday", 
    "to stay": "Tuesday",
    "tues day": "Tuesday",
    
    # Thursday variants
    "first day": "Thursday",
    "thirsty": "Thursday",
    "thursday's": "Thursday",
    "thirty": "Thursday",
    "thurs day": "Thursday",
    "the stay": "Thursday",
    "that's day": "Thursday",
    "fursday": "Thursday",
    "because day": "Thursday",
    "has day": "Thursday",
    "thus day": "Thursday",
    "does day": "Thursday",
    "as day": "Thursday",
    "still day": "Thursday",
    
    "when is day": "Wednesday",
    "winds day": "Wednesday",
    "wendsday": "Wednesday",
    
    "fry day": "Friday",
    "free day": "Friday",
    
    # Times
    "to pm": "2 PM",
    "too pm": "2 PM",
    "for pm": "4 PM",
    "free pm": "3 PM",
    "tin am": "10 AM",
    
    # Symptoms
    "to thick": "toothache",
    "tooth egg": "toothache",
    "two day": "toothache",
}

# Known Whisper hallucinations
HALLUCINATIONS = [
    "subtitles by", "amara.org", "thank you for watching", "thanks for watching",
    "please subscribe", "like and subscribe", "спасибо", "music", "♪",
    "transcribed by", "captioned by", "www.", ".com", ".org", "for watching",
]


def is_hallucination(text: str) -> bool:
    """Check if text is a known Whisper hallucination."""
    text_lower = text.lower()
    return any(h in text_lower for h in HALLUCINATIONS)


class ASRCorrector:
    """Corrects ASR errors using rule-based and LLM approaches."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self._correction_cache: Dict[str, str] = {}
        
    def correct(
        self,
        text: str,
        state: str = "",
        expected_words: List[str] = None,
        use_llm: bool = True,
    ) -> CorrectionResult:
        """Correct ASR transcription."""
        if not text.strip():
            return CorrectionResult(
                original=text, corrected=text, was_corrected=False,
                confidence=1.0, corrections_made=[],
            )
        
        corrections_made = []
        corrected = text
        
        # Stage 1: Check if it's a clear confirmation (before any other processing)
        if state in ["confirming_day", "confirming"]:
            confirm_type = self._is_clear_confirmation(corrected)
            if confirm_type in ["yes", "no"]:
                # Don't try to correct yes/no responses
                return CorrectionResult(
                    original=text, corrected=corrected.strip(),
                    was_corrected=False, confidence=1.0, corrections_made=[],
                )
        
        # Stage 2: Rule-based phonetic corrections
        corrected, rule_corrections = self._apply_phonetic_corrections(corrected, state)
        corrections_made.extend(rule_corrections)
        
        # Stage 3: Cleanup ASR artifacts
        corrected = self._cleanup_artifacts(corrected)
        
        # Stage 4: Context-aware correction based on state
        corrected, state_corrections = self._apply_state_corrections(corrected, state, expected_words)
        corrections_made.extend(state_corrections)
        
        # Stage 5: LLM correction ONLY for ambiguous day selection (not confirmation)
        if use_llm and self.llm and state == "collecting_day":
            if self._needs_llm_correction(corrected, state):
                llm_result = self._llm_correct(corrected, state, expected_words)
                if llm_result != corrected:
                    corrections_made.append(f"LLM: '{corrected}' -> '{llm_result}'")
                    corrected = llm_result
        
        was_corrected = corrected.lower().strip() != text.lower().strip()
        
        if was_corrected:
            logger.info(
                "asr_corrected",
                original=text[:50],
                corrected=corrected[:50],
                corrections=len(corrections_made),
            )
        
        return CorrectionResult(
            original=text,
            corrected=corrected.strip(),
            was_corrected=was_corrected,
            confidence=0.9 if was_corrected else 1.0,
            corrections_made=corrections_made,
        )
    
    def _is_clear_confirmation(self, text: str) -> str:
        """Check if text is a clear yes/no confirmation."""
        text_lower = text.lower().strip().rstrip('.!?')
        
        # Strong YES patterns
        yes_words = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "absolutely"]
        if text_lower in yes_words:
            return "yes"
        
        # Phrases with "yes" or "correct"
        if re.search(r'\b(yes|correct|right)\b', text_lower) and 'no' not in text_lower:
            return "yes"
        
        # Strong NO patterns
        no_words = ["no", "nope", "nah", "wrong", "incorrect"]
        if text_lower in no_words:
            return "no"
        
        return "unclear"
    
    def _apply_phonetic_corrections(self, text: str, state: str) -> tuple[str, List[str]]:
        """Apply rule-based phonetic corrections."""
        corrections = []
        result = text.lower()
        
        for wrong, right in PHONETIC_CORRECTIONS.items():
            if wrong in result:
                result = result.replace(wrong, right)
                corrections.append(f"'{wrong}' -> '{right}'")
        
        if text and text[0].isupper() and result:
            result = result[0].upper() + result[1:]
            
        return result, corrections
    
    def _cleanup_artifacts(self, text: str) -> str:
        """Remove common ASR artifacts."""
        patterns = [
            (r"\buh+\b", ""),
            (r"\bum+\b", ""),
            (r"\s+", " "),
        ]
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result.strip()
    
    def _apply_state_corrections(
        self, text: str, state: str, expected_words: List[str] = None
    ) -> tuple[str, List[str]]:
        """Apply corrections based on conversation state."""
        corrections = []
        result = text
        
        if state in ["collecting_day"]:
            result, day_corrections = self._correct_day_references(result)
            corrections.extend(day_corrections)
        elif state == "collecting_time":
            result, time_corrections = self._correct_time_references(result)
            corrections.extend(time_corrections)
        
        return result, corrections
    
    def _correct_day_references(self, text: str) -> tuple[str, List[str]]:
        """Correct day of week references."""
        corrections = []
        result = text.lower()
        
        day_patterns = [
            (r"\b(choose|shoes|to)\s*(day|stay)?\b", "Tuesday"),
            (r"\btues\w*\b", "Tuesday"),
            (r"\b(first|thirst|thir|fur|thus|because|has|does|still)\s*(day|stay)?\b", "Thursday"),
            (r"\bthurs\w*\b", "Thursday"),
            (r"\b(when\s*is|wind|wens|wednes)\s*(day)?\b", "Wednesday"),
            (r"\b(fry|free|fri)\s*(day)?\b", "Friday"),
            (r"\bmon\w*\b", "Monday"),
            (r"\bsat\w*\b", "Saturday"),
            (r"\bsun\w*\b", "Sunday"),
        ]
        
        for pattern, day in day_patterns:
            match = re.search(pattern, result)
            if match:
                if day.lower() not in result:
                    old = match.group(0)
                    result = re.sub(pattern, day, result)
                    corrections.append(f"Day: '{old}' -> '{day}'")
                    logger.debug("day_extracted", day=day, matched=match.group(0))
                    break  # Only apply first match
                else:
                    logger.debug("day_extracted", day=day, matched=day.lower())
                        
        return result, corrections
    
    def _correct_time_references(self, text: str) -> tuple[str, List[str]]:
        """Correct time references."""
        corrections = []
        result = text
        
        number_words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12",
        }
        
        for word, digit in number_words.items():
            pattern = rf"\b{word}\s*(am|pm|o'?clock)?\b"
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, f"{digit} \\1", result, flags=re.IGNORECASE)
                corrections.append(f"Time: '{word}' -> '{digit}'")
        
        return result, corrections
    
    def _needs_llm_correction(self, text: str, state: str) -> bool:
        """Determine if text needs LLM correction."""
        text_lower = text.lower().strip()
        
        if state == "collecting_day":
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if not any(day in text_lower for day in days):
                return len(text_lower) < 40
                
        return False
    
    def _llm_correct(self, text: str, state: str, expected_words: List[str] = None) -> str:
        """Use LLM to correct ambiguous transcription."""
        if not self.llm:
            return text
            
        cache_key = f"{state}:{text.lower()}"
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]
        
        if state == "collecting_day":
            prompt = f"""The user is selecting a day for an appointment. 
The ASR transcribed: "{text}"
The user probably said one of: Monday, Tuesday, Wednesday, Thursday, Friday.
"Thursday" often sounds like "still", "because", "has day", etc.
If this sounds like a day of the week, output just the day name.
If unclear, output: unclear
Output only one word."""
        else:
            return text
        
        try:
            from ..core.types import LLMMessage
            response = self.llm.generate(
                [LLMMessage(role="user", content=prompt)],
                max_tokens=20,
            )
            
            corrected = response.content.strip().strip('"').strip("'")
            
            # Validate
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "unclear"]
            if corrected.lower() not in days:
                corrected = text
            elif corrected.lower() == "unclear":
                corrected = text
            
            self._correction_cache[cache_key] = corrected
            logger.debug("llm_correction", original=text, corrected=corrected, state=state)
            return corrected
            
        except Exception as e:
            logger.error("llm_correction_failed", error=str(e))
            return text
