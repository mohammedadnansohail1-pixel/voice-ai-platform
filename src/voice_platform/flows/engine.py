"""Flow execution engine."""
import re
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum

from .models import Flow, FlowState, StateType, Intent
from ..logging import get_logger

logger = get_logger("flow_engine")


class EngineState(str, Enum):
    """Engine execution state."""
    IDLE = "idle"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    ENDED = "ended"


@dataclass
class FlowContext:
    """Runtime context for a flow execution."""
    flow_name: str
    current_state: str
    slots: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    retry_count: int = 0
    engine_state: EngineState = EngineState.IDLE
    
    def set_slot(self, name: str, value: Any) -> None:
        self.slots[name] = value
    
    def get_slot(self, name: str, default: Any = None) -> Any:
        return self.slots.get(name, default)
    
    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


@dataclass
class EngineResponse:
    """Response from the flow engine."""
    message: Optional[str] = None
    needs_input: bool = False
    ended: bool = False
    action_request: Optional[dict] = None
    current_state: str = ""
    slots: dict[str, Any] = field(default_factory=dict)


class FlowEngine:
    """Executes conversation flows."""
    
    def __init__(
        self,
        flow: Flow,
        intent_detector: Optional[Callable[[str, dict[str, Intent]], Optional[str]]] = None,
        action_executor: Optional[Callable[[str, dict], tuple[bool, Any]]] = None,
    ):
        self.flow = flow
        self.context: Optional[FlowContext] = None
        self._intent_detector = intent_detector or self._default_intent_detector
        self._action_executor = action_executor or self._default_action_executor
    
    def start(self) -> EngineResponse:
        """Start the flow from the initial state."""
        self.context = FlowContext(
            flow_name=self.flow.name,
            current_state=self.flow.initial_state,
            variables=dict(self.flow.context_defaults),
        )
        logger.info("flow_started", flow=self.flow.name, initial_state=self.flow.initial_state)
        return self._execute_current_state()
    
    def process_input(self, user_input: str) -> EngineResponse:
        """Process user input and advance the flow."""
        if not self.context:
            raise RuntimeError("Flow not started. Call start() first.")
        
        if self.context.engine_state == EngineState.ENDED:
            return EngineResponse(ended=True, current_state=self.context.current_state)
        
        self.context.add_history("user", user_input)
        logger.info("user_input", state=self.context.current_state, input=user_input)
        
        state = self._get_current_state()
        
        if state.type in (StateType.LISTEN, StateType.SPEAK_LISTEN):
            return self._handle_listen_input(state, user_input)
        else:
            return self._execute_current_state()
    
    def execute_action_result(self, success: bool, result: Any = None) -> EngineResponse:
        """Continue flow after external action completes."""
        if not self.context:
            raise RuntimeError("Flow not started.")
        
        state = self._get_current_state()
        
        if success and state.on_success:
            self.context.current_state = state.on_success
        elif not success and state.on_failure:
            self.context.current_state = state.on_failure
        
        if result is not None:
            self.context.variables["last_action_result"] = result
        
        return self._execute_current_state()
    
    def _get_current_state(self) -> FlowState:
        state_name = self.context.current_state
        if state_name not in self.flow.states:
            raise ValueError(f"Unknown state: {state_name}")
        return self.flow.states[state_name]
    
    def _execute_current_state(self) -> EngineResponse:
        state = self._get_current_state()
        logger.debug("executing_state", state=state.name, type=state.type.value)
        
        if state.type == StateType.SPEAK:
            return self._handle_speak(state)
        elif state.type == StateType.LISTEN:
            return self._handle_listen(state)
        elif state.type == StateType.SPEAK_LISTEN:
            return self._handle_speak_listen(state)
        elif state.type == StateType.ACTION:
            return self._handle_action(state)
        elif state.type == StateType.CONDITION:
            return self._handle_condition(state)
        elif state.type == StateType.END:
            return self._handle_end(state)
        else:
            raise ValueError(f"Unknown state type: {state.type}")
    
    def _handle_speak(self, state: FlowState) -> EngineResponse:
        """Handle a speak state."""
        message = self._interpolate(state.message or "")
        self.context.add_history("assistant", message)
        
        if state.next:
            self.context.current_state = state.next
            next_response = self._execute_current_state()
            
            return EngineResponse(
                message=message,
                needs_input=next_response.needs_input,
                ended=next_response.ended,
                action_request=next_response.action_request,
                current_state=self.context.current_state,
                slots=dict(self.context.slots),
            )
        
        return EngineResponse(
            message=message,
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _handle_listen(self, state: FlowState) -> EngineResponse:
        self.context.engine_state = EngineState.WAITING_INPUT
        return EngineResponse(
            needs_input=True,
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _handle_speak_listen(self, state: FlowState) -> EngineResponse:
        message = self._interpolate(state.message or "")
        self.context.add_history("assistant", message)
        self.context.engine_state = EngineState.WAITING_INPUT
        
        return EngineResponse(
            message=message,
            needs_input=True,
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _handle_action(self, state: FlowState) -> EngineResponse:
        action_name = state.action
        params = {k: self._interpolate(str(v)) for k, v in state.action_params.items()}
        
        return EngineResponse(
            action_request={"action": action_name, "params": params},
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _handle_condition(self, state: FlowState) -> EngineResponse:
        result = self._evaluate_condition(state.condition or "false")
        
        if result and state.if_true:
            self.context.current_state = state.if_true
        elif not result and state.if_false:
            self.context.current_state = state.if_false
        
        return self._execute_current_state()
    
    def _handle_end(self, state: FlowState) -> EngineResponse:
        self.context.engine_state = EngineState.ENDED
        message = self._interpolate(state.message or "")
        if message:
            self.context.add_history("assistant", message)
        
        logger.info("flow_ended", flow=self.flow.name, slots=self.context.slots)
        
        return EngineResponse(
            message=message if message else None,
            ended=True,
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _handle_listen_input(self, state: FlowState, user_input: str) -> EngineResponse:
        # Fill slots first
        for slot in state.slots:
            if slot.name not in self.context.slots:
                self.context.set_slot(slot.name, user_input)
                logger.info("slot_filled", slot=slot.name, value=user_input)
        
        # Detect intent
        intent_name = self._intent_detector(user_input, state.intents)
        
        if intent_name and intent_name in state.intents:
            intent = state.intents[intent_name]
            self.context.current_state = intent.next
            self.context.retry_count = 0
            logger.info("intent_matched", intent=intent_name, next_state=intent.next)
            return self._execute_current_state()
        
        # No intent matched - retry or fallback
        self.context.retry_count += 1
        
        if self.context.retry_count > state.max_retries:
            self.context.engine_state = EngineState.ENDED
            return EngineResponse(
                message=self.flow.global_fallback,
                ended=True,
                current_state=self.context.current_state,
                slots=dict(self.context.slots),
            )
        
        return EngineResponse(
            message=state.fallback_message,
            needs_input=True,
            current_state=self.context.current_state,
            slots=dict(self.context.slots),
        )
    
    def _interpolate(self, text: str) -> str:
        if not text:
            return text
        
        def replace_slot(match):
            slot_name = match.group(1)
            return str(self.context.get_slot(slot_name, f"[{slot_name}]"))
        
        text = re.sub(r'\{slots\.(\w+)\}', replace_slot, text)
        
        def replace_var(match):
            var_name = match.group(1)
            return str(self.context.variables.get(var_name, f"[{var_name}]"))
        
        text = re.sub(r'\{(\w+)\}', replace_var, text)
        
        return text
    
    def _evaluate_condition(self, condition: str) -> bool:
        match = re.match(r'slots\.(\w+)\s*==\s*["\']?(\w+)["\']?', condition)
        if match:
            slot_name, expected = match.groups()
            actual = str(self.context.get_slot(slot_name, ""))
            return actual.lower() == expected.lower()
        
        match = re.match(r'(\w+)\s*==\s*["\']?(\w+)["\']?', condition)
        if match:
            var_name, expected = match.groups()
            actual = str(self.context.variables.get(var_name, ""))
            return actual.lower() == expected.lower()
        
        return False
    
    def _default_intent_detector(self, user_input: str, intents: dict[str, Intent]) -> Optional[str]:
        """Default keyword-based intent detection."""
        user_lower = user_input.lower()
        
        # First pass: check for pattern matches
        for intent_name, intent in intents.items():
            if intent.patterns:  # Has patterns - check them
                for pattern in intent.patterns:
                    if pattern.lower() in user_lower:
                        return intent_name
        
        # Second pass: check for catch-all "any" intent (empty patterns)
        for intent_name, intent in intents.items():
            if not intent.patterns:  # Empty patterns = catch-all
                return intent_name
        
        return None
    
    def _default_action_executor(self, action: str, params: dict) -> tuple[bool, Any]:
        logger.warning("no_action_executor", action=action)
        return True, None
