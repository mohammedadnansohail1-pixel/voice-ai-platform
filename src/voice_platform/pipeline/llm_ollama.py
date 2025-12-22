"""Ollama LLM backend for intent classification and slot extraction."""
import json
import re
from typing import Optional
import requests

from .base import LLMBackend
from .registries import llm_registry
from ..logging import get_logger

logger = get_logger("llm.ollama")


@llm_registry.register("ollama")
class OllamaLLM(LLMBackend):
    """Ollama LLM backend."""
    
    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 15,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._loaded = False
    
    def load(self) -> None:
        if self._loaded:
            return
        
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            
            logger.info("warming_up_llm", model=self.model)
            self._generate_raw("Hi", "Say 'ready'")
            
            self._loaded = True
            logger.info("ollama_ready", model=self.model)
        except Exception as e:
            logger.error("ollama_connection_failed", error=str(e))
            raise
    
    def _generate_raw(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if not self._loaded:
            self.load()
        
        try:
            return self._generate_raw(prompt, system)
        except Exception as e:
            logger.error("llm_error", error=str(e))
            return ""
    
    def chat(self, messages: list[dict]) -> str:
        if not self._loaded:
            self.load()
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error("llm_error", error=str(e))
            return ""
    
    def classify_intent(self, user_input: str, intents: list[str], context: str = "") -> Optional[str]:
        """Classify user input into one of the given intents."""
        intent_list = ", ".join(intents)
        
        system = """You classify user intent for a medical appointment scheduling system.
Pay attention to NEGATION - if user says "can't", "don't", "not", consider the opposite meaning.
Reply with ONLY the intent name, nothing else."""
        
        prompt = f"""Available intents: {intent_list}

Context: {context}
User said: "{user_input}"

What is the user's intent? Reply with just the intent name:"""
        
        result = self.generate(prompt, system).strip().lower()
        result = result.replace('"', '').replace("'", "").strip()
        
        for intent in intents:
            if intent.lower() == result or intent.lower() in result.split():
                logger.debug("intent_classified", input=user_input[:30], intent=intent)
                return intent
        
        logger.debug("intent_not_matched", input=user_input[:30], result=result)
        return None
    
    def extract_slots(self, user_input: str, slots: dict[str, str], context: str = "") -> dict[str, str]:
        """Extract slot values from user input."""
        slot_list = "\n".join([f"- {k}: {v}" for k, v in slots.items()])
        
        system = """You extract AVAILABILITY information for appointment scheduling.

IMPORTANT: Understand the difference between when someone IS vs IS NOT available:
- "I work mornings" = NOT available mornings, so preferred_time should be "afternoon" or "evening"
- "I'm free mornings" = available mornings, so preferred_time = "morning"
- "I can't do Mondays" = NOT available Monday, don't set that as preferred
- "I work weekdays" = only available weekends, so preferred_day = "weekend"

Extract what times/days the user IS AVAILABLE, not when they're busy.
Return ONLY a JSON object. Use null for slots not found or unclear."""
        
        prompt = f"""Extract these slots (what the user IS AVAILABLE for):
{slot_list}

Context: {context}
User said: "{user_input}"

JSON:"""
        
        result = self.generate(prompt, system)
        
        try:
            json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                clean = {}
                for k, v in extracted.items():
                    if v and v != "null" and v != "None" and str(v).strip():
                        clean[k] = str(v).strip()
                logger.debug("slots_extracted", slots=clean)
                return clean
        except json.JSONDecodeError as e:
            logger.debug("slot_extraction_failed", error=str(e), result=result[:100])
        
        return {}
    
    def generate_response(self, user_input: str, context: str, instruction: str) -> str:
        prompt = f"""{instruction}
Context: {context}
User: "{user_input}"
Response (1 sentence):"""
        
        return self.generate(prompt, "Be brief and helpful.").strip()
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
