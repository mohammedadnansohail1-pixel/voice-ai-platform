"""
LLM-enhanced slot extraction for semantic understanding.

Uses LLM to understand implicit preferences that regex can't catch.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ...llm.ollama import OllamaLLM
from ...core.config import LLMConfig
from ...logging import get_logger

logger = get_logger("agent.llm_extractor")


@dataclass
class LLMExtraction:
    """Result from LLM extraction."""
    day: Optional[str] = None
    time: Optional[str] = None
    time_preference: Optional[str] = None  # "morning", "afternoon", "evening"
    confirmation: Optional[str] = None  # "yes", "no", "unclear"
    reason: Optional[str] = None
    raw_reasoning: Optional[str] = None


class LLMSlotExtractor:
    """
    Uses LLM to extract slots and understand implicit preferences.
    
    Examples:
        "I work in the morning" → time_preference: "afternoon"
        "Not too early please" → time_preference: "afternoon"  
        "After lunch works" → time_preference: "afternoon"
    """
    
    EXTRACTION_PROMPT = """You are extracting appointment booking information from user speech.

Current context:
- Collecting: {collecting}
- Available days: {available_days}
- Available times: {available_times}

User said: "{user_text}"

Extract the following (respond with ONLY the JSON, no explanation):
{{
    "day": null or day name like "Thursday",
    "time": null or specific time like "2:00 PM",
    "time_preference": null or "morning" or "afternoon",
    "confirmation": "yes" or "no" or "unclear",
    "reason": null or medical reason like "toothache"
}}

Rules:
- If user says they work/are busy in morning → time_preference: "afternoon"
- If user says they work/are busy in afternoon → time_preference: "morning"  
- If user says "after lunch", "later", "not early" → time_preference: "afternoon"
- If user says "early", "before lunch" → time_preference: "morning"
- Only set "day" or "time" if explicitly mentioned
- confirmation is "yes" for agreement, "no" for disagreement, "unclear" otherwise

JSON:"""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        if llm is None:
            config = LLMConfig(model="llama3.2:latest", temperature=0.1)
            self.llm = OllamaLLM(config)
        else:
            self.llm = llm
        
        logger.info("llm_slot_extractor_initialized")
    
    def extract(
        self,
        user_text: str,
        collecting: str,
        available_days: list[str],
        available_times: list[str],
    ) -> LLMExtraction:
        """
        Extract slots using LLM understanding.
        
        Args:
            user_text: What the user said
            collecting: What we're currently collecting ("day", "time", "reason")
            available_days: List of available days
            available_times: List of available times
            
        Returns:
            LLMExtraction with extracted values
        """
        prompt = self.EXTRACTION_PROMPT.format(
            collecting=collecting,
            available_days=", ".join(available_days),
            available_times=", ".join(available_times),
            user_text=user_text,
        )
        
        try:
            from ...core.types import LLMMessage
            messages = [LLMMessage(role="user", content=prompt)]
            response = self.llm.generate(messages, max_tokens=150).content
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                result = LLMExtraction(
                    day=data.get("day"),
                    time=data.get("time"),
                    time_preference=data.get("time_preference"),
                    confirmation=data.get("confirmation", "unclear"),
                    reason=data.get("reason"),
                    raw_reasoning=response,
                )
                
                logger.debug(
                    "llm_extraction_complete",
                    user_text=user_text[:50],
                    day=result.day,
                    time=result.time,
                    time_preference=result.time_preference,
                    confirmation=result.confirmation,
                )
                
                return result
            else:
                logger.warning("llm_extraction_no_json", response=response[:100])
                return LLMExtraction(confirmation="unclear")
                
        except Exception as e:
            logger.error("llm_extraction_failed", error=str(e))
            return LLMExtraction(confirmation="unclear")
