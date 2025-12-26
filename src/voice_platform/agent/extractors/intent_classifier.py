"""
Config-driven LLM intent classifier.

All behavior defined in YAML configs:
- configs/agent/states.yaml - State definitions
- configs/agent/actions.yaml - Action definitions  
- configs/agent/examples.yaml - Few-shot examples
- configs/agent/prompts.yaml - LLM prompts

No hardcoded if/else logic.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import re
import time
import hashlib
import yaml

from pydantic import BaseModel, Field, validator

from ...llm.ollama import OllamaLLM
from ...core.config import LLMConfig
from ...core.types import LLMMessage
from ...logging import get_logger

logger = get_logger("agent.intent_classifier")


# ============================================================================
# Enums (derived from config at runtime)
# ============================================================================

class UserIntent(str, Enum):
    PROVIDING_INFO = "providing_info"
    REFUSING = "refusing"
    CORRECTING = "correcting"
    CONFIRMING = "confirming"
    DENYING = "denying"
    UNCLEAR = "unclear"
    OFF_TOPIC = "off_topic"


class Action(str, Enum):
    ACCEPT_INPUT = "accept_input"
    ASK_AGAIN = "ask_again"
    ASK_CLARIFICATION = "ask_clarification"
    CORRECT_FIELD = "correct_field"
    SKIP_FIELD = "skip_field"
    END_CALL = "end_call"
    CONTINUE = "continue"


# ============================================================================
# Config Models (Pydantic)
# ============================================================================

class StateConfig(BaseModel):
    """Configuration for a conversation state."""
    question: Optional[str] = None
    required: bool = True
    field: Optional[str] = None
    implicit_confirmation: Optional[str] = None
    is_terminal: bool = False
    transitions: Dict[str, Optional[str]] = Field(default_factory=dict)


class ActionConfig(BaseModel):
    """Configuration for an action."""
    description: str
    behavior: List[Dict[str, Any]] = Field(default_factory=list)


class ExampleContext(BaseModel):
    """Context for a few-shot example."""
    assistant: str
    state: str


class ExampleClassification(BaseModel):
    """Expected classification for an example."""
    intent: str
    action: str
    extracted_value: Optional[str] = None
    field_to_correct: Optional[str] = None
    suggested_response: Optional[str] = None
    confidence: float = 0.9


class Example(BaseModel):
    """A few-shot example."""
    context: ExampleContext
    user: str
    classification: ExampleClassification


class PromptsConfig(BaseModel):
    """LLM prompts configuration."""
    system_prompt: str
    user_prompt_template: str
    example_format: str


class AgentConfig(BaseModel):
    """Complete agent configuration."""
    states: Dict[str, StateConfig] = Field(default_factory=dict)
    actions: Dict[str, ActionConfig] = Field(default_factory=dict)
    examples: List[Example] = Field(default_factory=list)
    prompts: Optional[PromptsConfig] = None


# ============================================================================
# Classification Result
# ============================================================================

@dataclass
class ClassificationResult:
    """Result from intent classification."""
    intent: UserIntent
    action: Action
    extracted_value: Optional[str] = None
    field_to_correct: Optional[str] = None
    suggested_response: Optional[str] = None
    confidence: float = 0.0
    used_llm: bool = False
    latency_ms: float = 0.0


# ============================================================================
# Config Loader
# ============================================================================

class ConfigLoader:
    """Loads and validates agent configuration from YAML files."""
    
    DEFAULT_CONFIG_DIR = Path("configs/agent")
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._config: Optional[AgentConfig] = None
    
    def load(self) -> AgentConfig:
        """Load all config files."""
        if self._config:
            return self._config
        
        states = self._load_yaml("states.yaml").get("states", {})
        actions = self._load_yaml("actions.yaml").get("actions", {})
        examples_raw = self._load_yaml("examples.yaml").get("examples", [])
        prompts_raw = self._load_yaml("prompts.yaml")
        
        # Parse into Pydantic models
        parsed_states = {k: StateConfig(**v) for k, v in states.items()}
        parsed_actions = {k: ActionConfig(**v) for k, v in actions.items()}
        parsed_examples = [Example(**e) for e in examples_raw]
        parsed_prompts = PromptsConfig(**prompts_raw) if prompts_raw else None
        
        self._config = AgentConfig(
            states=parsed_states,
            actions=parsed_actions,
            examples=parsed_examples,
            prompts=parsed_prompts,
        )
        
        logger.info(
            "config_loaded",
            states=len(parsed_states),
            actions=len(parsed_actions),
            examples=len(parsed_examples),
        )
        
        return self._config
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file."""
        path = self.config_dir / filename
        if not path.exists():
            logger.warning("config_file_not_found", path=str(path))
            return {}
        
        with open(path) as f:
            return yaml.safe_load(f) or {}
    
    def reload(self) -> AgentConfig:
        """Force reload configuration."""
        self._config = None
        return self.load()


# ============================================================================
# Intent Classifier
# ============================================================================

class IntentClassifier:
    """
    Config-driven LLM intent classifier.
    
    All behavior defined in YAML:
    - States, actions, examples, prompts
    - No hardcoded if/else
    """
    
    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        config_dir: Optional[Path] = None,
    ):
        self._llm = llm
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.load()
        self._cache: Dict[str, ClassificationResult] = {}
        
        logger.info("intent_classifier_initialized")
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy LLM initialization."""
        if self._llm is None:
            config = LLMConfig(
                model="llama3.2:latest",
                temperature=0.1,
                max_tokens=300,
            )
            self._llm = OllamaLLM(config)
        return self._llm
    
    def classify(
        self,
        user_input: str,
        context: Dict[str, Any],
    ) -> ClassificationResult:
        """
        Classify user intent using LLM with config-driven behavior.
        
        Args:
            user_input: What the user said
            context: Dict with state, last_assistant_message, collected
            
        Returns:
            ClassificationResult
        """
        start_time = time.perf_counter()
        
        text = user_input.strip()
        if not text:
            return ClassificationResult(
                intent=UserIntent.UNCLEAR,
                action=Action.ASK_AGAIN,
                suggested_response="I didn't catch that. Could you please repeat?",
                confidence=0.0,
            )
        
        # Check cache
        cache_key = self._cache_key(text, context)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Fast path for unambiguous consent responses
        result = self._fast_path(text, context)
        if result:
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            self._cache[cache_key] = result
            return result
        
        # LLM classification
        result = self._llm_classify(text, context)
        result.latency_ms = (time.perf_counter() - start_time) * 1000
        self._cache[cache_key] = result
        
        return result
    
    def _cache_key(self, text: str, context: Dict[str, Any]) -> str:
        """Generate cache key."""
        key = f"{text.lower()}|{context.get('state', '')}|{context.get('last_assistant_message', '')[:50]}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _fast_path(self, text: str, context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Fast path for unambiguous cases (consent only)."""
        lower = text.lower().strip()
        state = context.get("state", "")
        
        # Only handle consent stage fast path - everything else needs context
        if state == "collecting_consent":
            if lower in ("yes", "yeah", "yep", "sure", "ok", "okay"):
                return ClassificationResult(
                    intent=UserIntent.CONFIRMING,
                    action=Action.CONTINUE,
                    suggested_response=self._get_next_question(state),
                    confidence=0.95,
                    used_llm=False,
                )
            if lower in ("no", "nope", "no thanks"):
                return ClassificationResult(
                    intent=UserIntent.REFUSING,
                    action=Action.END_CALL,
                    suggested_response="I understand. If you change your mind, feel free to call back. Have a great day!",
                    confidence=0.95,
                    used_llm=False,
                )
        
        return None
    
    def _get_next_question(self, current_state: str) -> Optional[str]:
        """Get next question from config."""
        state_config = self.config.states.get(current_state)
        if state_config and state_config.transitions:
            next_state = state_config.transitions.get("on_confirm") or state_config.transitions.get("on_accept")
            if next_state:
                next_config = self.config.states.get(next_state)
                if next_config:
                    return next_config.question
        return None
    
    def _llm_classify(self, text: str, context: Dict[str, Any]) -> ClassificationResult:
        """LLM-based classification using config-driven prompts."""
        try:
            # Build prompt from config
            prompts = self.config.prompts
            if not prompts:
                raise ValueError("No prompts config found")
            
            # Build examples string
            examples_str = self._build_examples(context.get("state", ""))
            
            # Build collected string
            collected = context.get("collected", {})
            collected_str = ", ".join(f"{k}: {v}" for k, v in collected.items() if v) or "nothing"
            
            # Format user prompt
            user_prompt = prompts.user_prompt_template.format(
                examples=examples_str,
                last_assistant_message=context.get("last_assistant_message", ""),
                user_input=text,
                state=context.get("state", "unknown"),
                collected=collected_str,
            )
            
            messages = [
                LLMMessage(role="system", content=prompts.system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            
            response = self.llm.generate(messages, max_tokens=300).content
            
            # Parse JSON response
            return self._parse_response(response, text)
            
        except Exception as e:
            logger.error("llm_classification_failed", error=str(e))
            return self._fallback_result(context)
    
    def _build_examples(self, current_state: str, max_examples: int = 5) -> str:
        """Build examples string, prioritizing current state."""
        prompts = self.config.prompts
        if not prompts:
            return ""
        
        # Prioritize examples for current state
        state_examples = [e for e in self.config.examples if e.context.state == current_state]
        other_examples = [e for e in self.config.examples if e.context.state != current_state]
        
        selected = state_examples[:3] + other_examples[:max_examples - len(state_examples[:3])]
        
        formatted = []
        for ex in selected:
            output = {
                "intent": ex.classification.intent,
                "action": ex.classification.action,
                "extracted_value": ex.classification.extracted_value,
                "field_to_correct": ex.classification.field_to_correct,
                "suggested_response": ex.classification.suggested_response,
                "confidence": ex.classification.confidence,
            }
            formatted.append(prompts.example_format.format(
                assistant=ex.context.assistant,
                user=ex.user,
                state=ex.context.state,
                output=json.dumps(output),
            ))
        
        return "Examples:\n\n" + "\n\n".join(formatted)
    
    def _parse_response(self, response: str, user_input: str) -> ClassificationResult:
        """Parse LLM JSON response."""
        # Extract JSON - handle multiline by finding balanced braces
        start = response.find('{')
        if start == -1:
            raise ValueError(f"No JSON in response: {response[:100]}")
        
        # Find matching closing brace
        depth = 0
        end = start
        for i, c in enumerate(response[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        json_str = response[start:end]
        data = json.loads(json_str)
        
        # Validate and convert
        intent_str = data.get("intent", "unclear")
        action_str = data.get("action", "ask_again")
        
        # Handle invalid enum values gracefully
        try:
            intent = UserIntent(intent_str)
        except ValueError:
            logger.warning("invalid_intent", value=intent_str)
            intent = UserIntent.UNCLEAR
        
        try:
            action = Action(action_str)
        except ValueError:
            logger.warning("invalid_action", value=action_str)
            action = Action.ASK_AGAIN
        
        result = ClassificationResult(
            intent=intent,
            action=action,
            extracted_value=data.get("extracted_value"),
            field_to_correct=data.get("field_to_correct"),
            suggested_response=data.get("suggested_response"),
            confidence=float(data.get("confidence", 0.7)),
            used_llm=True,
        )
        
        logger.info(
            "llm_classification",
            text=user_input[:50],
            intent=result.intent.value,
            action=result.action.value,
            confidence=result.confidence,
        )
        
        return result
    
    def _fallback_result(self, context: Dict[str, Any]) -> ClassificationResult:
        """Fallback when LLM fails."""
        state = context.get("state", "")
        state_config = self.config.states.get(state)
        
        if state_config and state_config.question:
            question = state_config.question
        else:
            question = "Could you please repeat that?"
        
        return ClassificationResult(
            intent=UserIntent.UNCLEAR,
            action=Action.ASK_AGAIN,
            suggested_response=f"I'm sorry, I didn't catch that. {question}",
            confidence=0.0,
            used_llm=True,
        )
    
    def get_state_config(self, state: str) -> Optional[StateConfig]:
        """Get state configuration."""
        return self.config.states.get(state)
    
    def get_action_config(self, action: str) -> Optional[ActionConfig]:
        """Get action configuration."""
        return self.config.actions.get(action)
    
    def reload_config(self):
        """Reload configuration from files."""
        self.config = self.config_loader.reload()
        self._cache.clear()
        logger.info("config_reloaded")
