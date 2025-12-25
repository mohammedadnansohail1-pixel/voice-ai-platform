"""
N-best + Phonetic Matching ASR Correction.

Industry-standard approach: phonetic similarity instead of hardcoded mappings.
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re

import jellyfish

from ..logging import get_logger

logger = get_logger("asr.phonetic")


@dataclass
class CorrectionResult:
    """Result of ASR correction."""
    original: str
    corrected: str
    was_corrected: bool
    confidence: float
    method: str
    match_score: float


class PhoneticMatcher:
    """
    Phonetic similarity matcher using multiple algorithms.
    """
    
    MIN_SIMILARITY = 0.60  # Threshold for accepting a match
    PHONETIC_BOOST = 0.12  # Boost for matching phonetic codes
    
    def find_best_match(
        self,
        text: str,
        candidates: List[str],
    ) -> Optional[Tuple[str, float]]:
        """Find the candidate that best matches the text phonetically."""
        if not text or not candidates:
            return None
            
        text_lower = text.lower().strip()
        text_words = text_lower.split()
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Strategy 1: Exact substring match
            if candidate_lower in text_lower:
                return (candidate, 1.0)
            
            # Strategy 2: Full string Jaro-Winkler
            score = jellyfish.jaro_winkler_similarity(text_lower, candidate_lower)
            
            # Strategy 3: Match each word individually
            for word in text_words:
                word_score = jellyfish.jaro_winkler_similarity(word, candidate_lower)
                
                # Boost if phonetic codes match
                try:
                    if jellyfish.soundex(word) == jellyfish.soundex(candidate_lower):
                        word_score += self.PHONETIC_BOOST
                    if jellyfish.metaphone(word) == jellyfish.metaphone(candidate_lower):
                        word_score += self.PHONETIC_BOOST
                except:
                    pass
                
                if word_score > score:
                    score = word_score
            
            # Strategy 4: First word might be the garbled day name
            if text_words:
                first_word = text_words[0]
                first_score = jellyfish.jaro_winkler_similarity(first_word, candidate_lower)
                if first_score > score:
                    score = first_score
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= self.MIN_SIMILARITY:
            return (best_match, best_score)
        
        return None


class PhoneticCorrector:
    """
    ASR correction using phonetic matching.
    
    Replaces hardcoded PHONETIC_CORRECTIONS with generalizable algorithm.
    """
    
    def __init__(self, llm=None):
        self.matcher = PhoneticMatcher()
        self.llm = llm
        
        # State-to-vocabulary mapping
        self.state_vocabulary = {
            "collecting_day": [
                "Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday"
            ],
            "confirming_day": [
                "Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday",
                "yes", "no", "correct", "wrong"
            ],
            "collecting_time": [
                "9 AM", "10 AM", "11 AM", "2 PM", "3 PM", "4 PM",
                "nine", "ten", "eleven", "two", "three", "four"
            ],
            "confirming": ["yes", "no", "correct", "wrong", "cancel"],
        }
        
        # Don't modify clear confirmations
        self.confirmation_patterns = [
            r"^(yes|yeah|yep|yup|sure|ok|okay|correct|right|absolutely)\.?!?$",
            r"^(no|nope|nah|wrong|incorrect|cancel)\.?!?$",
        ]
    
    def correct(
        self,
        text: str,
        state: str = "",
        expected_words: List[str] = None,
        use_llm: bool = False,
    ) -> CorrectionResult:
        """Correct ASR transcription using phonetic matching."""
        original = text.strip()
        
        if not original:
            return CorrectionResult(
                original=text, corrected=text, was_corrected=False,
                confidence=1.0, method="none", match_score=0.0
            )
        
        # Don't modify clear confirmations
        if self._is_confirmation(original):
            return CorrectionResult(
                original=text, corrected=original, was_corrected=False,
                confidence=1.0, method="confirmation", match_score=1.0
            )
        
        # Get vocabulary for this state
        candidates = expected_words or self.state_vocabulary.get(state, [])
        
        if not candidates:
            return CorrectionResult(
                original=text, corrected=original, was_corrected=False,
                confidence=1.0, method="no_candidates", match_score=0.0
            )
        
        # Check for exact match first
        for candidate in candidates:
            if candidate.lower() in original.lower():
                return CorrectionResult(
                    original=text, corrected=original, was_corrected=False,
                    confidence=1.0, method="exact", match_score=1.0
                )
        
        # Try phonetic matching
        match_result = self.matcher.find_best_match(original, candidates)
        
        if match_result:
            matched_word, score = match_result
            logger.info(
                "phonetic_match",
                original=original,
                matched=matched_word,
                score=f"{score:.2f}",
                state=state,
            )
            return CorrectionResult(
                original=text, corrected=matched_word, was_corrected=True,
                confidence=score, method="phonetic", match_score=score
            )
        
        # LLM fallback for short ambiguous inputs
        if use_llm and self.llm and len(original) < 30:
            llm_result = self._llm_correct(original, state, candidates)
            if llm_result:
                logger.info(
                    "llm_correction",
                    original=original,
                    corrected=llm_result,
                    state=state,
                )
                return CorrectionResult(
                    original=text, corrected=llm_result, was_corrected=True,
                    confidence=0.7, method="llm", match_score=0.0
                )
        
        # No correction
        return CorrectionResult(
            original=text, corrected=original, was_corrected=False,
            confidence=0.5, method="none", match_score=0.0
        )
    
    def _is_confirmation(self, text: str) -> bool:
        """Check if text is a clear yes/no."""
        text_lower = text.lower().strip()
        for pattern in self.confirmation_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _llm_correct(self, text: str, state: str, candidates: List[str]) -> Optional[str]:
        """LLM fallback for disambiguation."""
        if not self.llm:
            return None
        
        what = state.replace("collecting_", "").replace("confirming_", "").replace("_", " ")
        prompt = f"""User is selecting a {what}. ASR heard: "{text}"
Options: {', '.join(candidates[:7])}
If it sounds like an option, output just that word. Otherwise output: UNCLEAR"""

        try:
            from ..core.types import LLMMessage
            response = self.llm.generate(
                [LLMMessage(role="user", content=prompt)],
                max_tokens=20,
            )
            result = response.content.strip().strip('"\'.,')
            
            if result.upper() == "UNCLEAR":
                return None
            
            # Validate against candidates
            for c in candidates:
                if result.lower() == c.lower():
                    return c
            
            return None
        except Exception as e:
            logger.warning("llm_correction_failed", error=str(e))
            return None
