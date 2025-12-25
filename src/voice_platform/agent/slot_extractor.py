"""
Rule-based slot extraction for healthcare appointment scheduling.

Key principle: LLMs hallucinate, regex doesn't.
Includes fuzzy matching for common ASR errors.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from ..logging import get_logger

logger = get_logger("agent.slot_extractor")


class ConfirmationType(Enum):
    """Type of confirmation response."""
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


@dataclass
class ExtractedSlots:
    """Container for extracted appointment slots."""
    visit_reason: Optional[str] = None
    preferred_day: Optional[str] = None
    preferred_time: Optional[str] = None
    confirmation: ConfirmationType = ConfirmationType.UNCLEAR
    raw_text: str = ""
    newly_extracted: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "visit_reason": self.visit_reason,
            "preferred_day": self.preferred_day,
            "preferred_time": self.preferred_time,
            "confirmation": self.confirmation.value,
        }
    
    def is_complete(self) -> bool:
        return all([self.visit_reason, self.preferred_day, self.preferred_time])
    
    def missing_slots(self) -> List[str]:
        missing = []
        if not self.visit_reason:
            missing.append("visit_reason")
        if not self.preferred_day:
            missing.append("preferred_day")
        if not self.preferred_time:
            missing.append("preferred_time")
        return missing


class SlotExtractor:
    """Rule-based slot extractor with fuzzy matching for ASR errors."""
    
    # Day patterns - includes common ASR mistakes
    DAY_PATTERNS: Dict[str, List[str]] = {
        "Monday": ["monday", "mon", "monda"],
        "Tuesday": [
            "tuesday", "tue", "tues", "tuseday",
            "choose day", "shoes day",  # ASR mistakes
        ],
        "Wednesday": [
            "wednesday", "wed", "weds", "wendsday", "wensday",
            "when is day", "winds day",  # ASR mistakes
        ],
        "Thursday": [
            "thursday", "thu", "thur", "thurs", "thursdy",
            "thurs day", "first day", "thirsty", "thirty",  # ASR mistakes
            "that's day", "the stay", "thursday's",
        ],
        "Friday": [
            "friday", "fri", "fridy",
            "fry day", "free day",  # ASR mistakes
        ],
        "Saturday": ["saturday", "sat", "saterday"],
        "Sunday": ["sunday", "sun", "sonday"],
    }
    
    # Additional pattern: "X works" where X sounds like a day
    DAY_WORKS_PATTERNS = [
        (r"\b(tues|choose|shoes)\w*\s+works", "Tuesday"),
        (r"\b(wed|when|wind)\w*\s+works", "Wednesday"),
        (r"\b(thur|thir|first|that)\w*\s+works", "Thursday"),
        (r"\b(fri|fry|free)\w*\s+works", "Friday"),
    ]
    
    # Negative context - don't extract time from these
    NEGATIVE_TIME_CONTEXT = [
        r"i\s+work\s+(?:in\s+)?(?:the\s+)?",
        r"i(?:'m|\s+am)\s+(?:busy|working|not\s+free|not\s+available)\s+(?:in\s+)?(?:the\s+)?",
        r"can(?:'t|not)\s+(?:do|make)\s+(?:the\s+)?",
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        (r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm|a\.m\.|p\.m\.)", 
         lambda m: f"{int(m.group(1))}:{m.group(2)} {m.group(3).upper().replace('.', '')}"),
        (r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", 
         lambda m: f"{int(m.group(1))}:00 {m.group(2).upper().replace('.', '')}"),
        (r"(\d{1,2})\s*(?:o'?clock\s*)?(?:in\s+the\s+)?(morning)", 
         lambda m: f"{int(m.group(1))}:00 AM"),
        (r"(\d{1,2})\s*(?:o'?clock\s*)?(?:in\s+the\s+)?(afternoon|evening)", 
         lambda m: f"{int(m.group(1))}:00 PM"),
        (r"(?:at\s+)?(\d{1,2})\s*o'?clock", 
         lambda m: f"{int(m.group(1))}:00 PM" if int(m.group(1)) < 7 else f"{int(m.group(1))}:00 AM"),
    ]
    
    # Reason patterns
    REASON_PATTERNS = [
        (r"(tooth\s*ache|tooth\s*pain|teeth\s*hurt|tooth\s*hurt)", "toothache"),
        (r"(cavity|cavities)", "cavity"),
        (r"(cleaning|teeth\s*clean)", "cleaning"),
        (r"(root\s*canal)", "root canal"),
        (r"(crown)", "crown"),
        (r"(filling)", "filling"),
        (r"(check\s*up|checkup|physical|annual|routine)", "checkup"),
        (r"(head\s*ache|headache|migraine)", "headache"),
        (r"(back\s*ache|back\s*pain)", "back pain"),
        (r"(fever|temperature)", "fever"),
        (r"(cough|cold|flu|sick)", "cold/flu"),
        (r"(sore\s*throat)", "sore throat"),
        (r"(follow\s*up|followup)", "follow-up"),
        (r"(consultation|consult)", "consultation"),
        (r"(pain|hurt|ache)", "pain"),
    ]
    
    # YES patterns
    YES_PATTERNS = [
        r"^\s*(yes|yeah|yep|yup|sure|ok|okay|correct|right|absolutely|definitely|please|perfect|great|sounds?\s*good)\s*[.!]?\s*$",
        r"^(y|ye|ya)$",
        r"(?:yes|yeah),?\s*(?:please|book|that)",
        r"(?:let'?s?|we.?ll?)\s*(?:do|go\s*with)\s*(?:it|that)",
        r"book\s*(?:it|that)",
        r"that'?s?\s*(?:correct|right|it)$",
    ]
    
    # NO patterns
    NO_PATTERNS = [
        r"^\s*(no|nope|nah|cancel|wrong|incorrect|never\s*mind)\s*[.!]?\s*$",
        r"^no,?\s",
        r"\bthat'?s?\s*(wrong|incorrect|not\s*right)\b",
        r"\bcancel\s*(it|that|the)?\b",
        r"\bstart\s*over\b",
    ]

    def __init__(self):
        logger.info("slot_extractor_initialized")

    def extract(self, text: str) -> ExtractedSlots:
        """Extract all slots from user input."""
        result = ExtractedSlots(raw_text=text)
        text_lower = text.lower().strip()
        
        # Extract day (with fuzzy matching)
        day = self._extract_day(text_lower)
        if day:
            result.preferred_day = day
            result.newly_extracted.append("preferred_day")
        
        # Extract time
        time = self._extract_time(text_lower)
        if time:
            result.preferred_time = time
            result.newly_extracted.append("preferred_time")
        
        # Extract reason
        reason = self._extract_reason(text_lower)
        if reason:
            result.visit_reason = reason
            result.newly_extracted.append("visit_reason")
        
        # Extract confirmation
        result.confirmation = self._extract_confirmation(text_lower)
        
        logger.debug(
            "slots_extracted",
            day=result.preferred_day,
            time=result.preferred_time,
            reason=result.visit_reason,
            confirmation=result.confirmation.value,
            newly_extracted=result.newly_extracted,
        )
        
        return result

    def _extract_day(self, text: str) -> Optional[str]:
        """Extract day of week with fuzzy matching for ASR errors."""
        # First try exact patterns
        for canonical, variants in self.DAY_PATTERNS.items():
            for variant in variants:
                if re.search(rf"\b{re.escape(variant)}\b", text):
                    logger.debug("day_extracted", day=canonical, matched=variant)
                    return canonical
        
        # Try "X works" patterns for ASR errors
        for pattern, day in self.DAY_WORKS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug("day_extracted_fuzzy", day=day, pattern=pattern)
                return day
        
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time - context-aware."""
        for neg_pattern in self.NEGATIVE_TIME_CONTEXT:
            if re.search(neg_pattern, text, re.IGNORECASE):
                logger.debug("time_extraction_blocked", reason="negative_context", text=text[:30])
                return None
        
        for pattern, formatter in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        
        return None

    def _extract_reason(self, text: str) -> Optional[str]:
        """Extract visit reason from text."""
        for pattern, canonical in self.REASON_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return canonical
        return None

    def _extract_confirmation(self, text: str) -> ConfirmationType:
        """Extract yes/no confirmation."""
        # Skip questions
        if text.strip().endswith("?"):
            return ConfirmationType.UNCLEAR
        
        # Skip question patterns
        if re.search(r"\b(do you|don'?t you|can you|could you|would you)\b", text, re.IGNORECASE):
            return ConfirmationType.UNCLEAR
        
        # Skip polite phrases with "no"
        if re.search(r"\b(no worries|no problem|no rush|not a problem)\b", text, re.IGNORECASE):
            return ConfirmationType.UNCLEAR
        
        # Check YES
        for pattern in self.YES_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return ConfirmationType.YES
        
        # Check NO
        for pattern in self.NO_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return ConfirmationType.NO
        
        return ConfirmationType.UNCLEAR
