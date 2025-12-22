"""Flow execution engine with LLM fallback."""
from typing import Optional, Any
from dataclasses import dataclass, field

from .models import Flow, FlowState, StateType
from ..logging import get_logger

logger = get_logger("flows.engine")


@dataclass
class FlowContext:
    """Runtime context for flow execution."""
    current_state: str
    slots: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class EngineResponse:
    """Response from the flow engine."""
    message: Optional[str] = None
    needs_input: bool = False
    ended: bool = False
    action_request: Optional[str] = None
    current_state: str = ""
    slots: dict[str, Any] = field(default_factory=dict)


class FlowEngine:
    """Execute conversation flows with LLM-first approach."""
    
    def __init__(self, flow: Flow, llm=None):
        self.flow = flow
        self.context = FlowContext(
            current_state=flow.initial_state,
            variables=dict(flow.context_defaults),
        )
        self.llm = llm
        self._pending_action: Optional[str] = None
    
    def set_llm(self, llm) -> None:
        self.llm = llm
    
    def start(self) -> EngineResponse:
        logger.info("flow_started", flow=self.flow.name, initial_state=self.context.current_state)
        return self._execute_current_state()
    
    def process_input(self, user_input: str) -> EngineResponse:
        state = self._get_current_state()
        
        if not state:
            return EngineResponse(ended=True, current_state=self.context.current_state)
        
        logger.info("user_input", input=user_input, state=self.context.current_state)
        self.context.history.append({"role": "user", "content": user_input})
        
        # Extract slots with LLM FIRST
        if self.llm:
            extracted_slots = self._extract_slots_with_llm(user_input, state)
            for slot_name, value in extracted_slots.items():
                if value:
                    self.context.slots[slot_name] = value
                    logger.info("slot_filled", slot=slot_name, value=value)
        
        # Match intent (LLM first, then keywords)
        matched_intent, next_state = self._match_intent(user_input, state)
        
        if matched_intent:
            logger.info("intent_matched", intent=matched_intent, next_state=next_state)
            self.context.retry_count = 0
            self.context.current_state = next_state
            return self._execute_current_state()
        
        # No match - handle retry
        self.context.retry_count += 1
        
        if self.context.retry_count >= state.max_retries:
            logger.info("max_retries_reached", fallback=self.flow.global_fallback)
            return EngineResponse(
                message=self.flow.global_fallback,
                ended=True,
                current_state=self.context.current_state,
                slots=self.context.slots,
            )
        
        return EngineResponse(
            message=self._interpolate(state.fallback_message),
            needs_input=True,
            current_state=self.context.current_state,
            slots=self.context.slots,
        )
    
    def _match_intent(self, user_input: str, state: FlowState) -> tuple[Optional[str], Optional[str]]:
        """Match user input to an intent. LLM first, then keywords."""
        
        # Get available intents (excluding "any")
        intent_names = [name for name in state.intents.keys() if name != "any"]
        
        # Try LLM classification FIRST if available and we have intents
        if self.llm and intent_names:
            llm_intent = self.llm.classify_intent(
                user_input,
                intent_names,
                context=self._get_context_summary()
            )
            if llm_intent and llm_intent in state.intents:
                return llm_intent, state.intents[llm_intent].next
        
        # Fall back to keyword matching (only if LLM didn't match)
        if not self.llm:
            user_lower = user_input.lower()
            
            # Check for negation - skip keyword match if user says "can't", "don't", "not"
            has_negation = any(neg in user_lower for neg in ["can't", "cannot", "don't", "not ", "no "])
            
            if not has_negation:
                for intent_name, intent in state.intents.items():
                    if intent_name == "any":
                        continue
                    for pattern in intent.patterns:
                        if pattern.lower() in user_lower:
                            return intent_name, intent.next
        
        # Check for catch-all "any" intent
        if "any" in state.intents:
            return "any", state.intents["any"].next
        
        return None, None
    
    def _extract_slots_with_llm(self, user_input: str, state: FlowState) -> dict[str, str]:
        """Extract slots using LLM."""
        if not self.llm:
            return {}
        
        # Build slot descriptions
        slots_to_extract = {}
        
        # From current state
        for slot in state.slots:
            slots_to_extract[slot.name] = slot.prompt or slot.name.replace("_", " ")
        
        # Common appointment slots if not already filled
        common_slots = {
            "preferred_day": "day of the week (Monday, Tuesday, etc.) or 'weekday'/'weekend'",
            "preferred_time": "time of day (morning, afternoon, evening) or specific time",
            "visit_reason": "reason for the visit or medical issue",
        }
        
        for name, desc in common_slots.items():
            if name not in self.context.slots:
                slots_to_extract[name] = desc
        
        if not slots_to_extract:
            return {}
        
        return self.llm.extract_slots(
            user_input,
            slots_to_extract,
            context=self._get_context_summary()
        )
    
    def _get_context_summary(self) -> str:
        parts = []
        
        if self.context.slots:
            slot_str = ", ".join([f"{k}={v}" for k, v in self.context.slots.items()])
            parts.append(f"Collected: {slot_str}")
        
        recent = self.context.history[-4:]
        if recent:
            history_str = " | ".join([f"{h['role']}: {h['content'][:40]}" for h in recent])
            parts.append(f"Recent: {history_str}")
        
        return " ".join(parts) if parts else "New conversation"
    
    def execute_action_result(self, success: bool, data: Optional[dict] = None) -> EngineResponse:
        state = self._get_current_state()
        if not state:
            return EngineResponse(ended=True, current_state=self.context.current_state)
        
        self._pending_action = None
        
        if success and state.on_success:
            self.context.current_state = state.on_success
        elif not success and state.on_failure:
            self.context.current_state = state.on_failure
        elif state.next:
            self.context.current_state = state.next
        
        return self._execute_current_state()
    
    def _execute_current_state(self) -> EngineResponse:
        state = self._get_current_state()
        
        if not state:
            logger.info("flow_ended", flow=self.flow.name, slots=self.context.slots)
            return EngineResponse(ended=True, current_state=self.context.current_state, slots=self.context.slots)
        
        if state.type == StateType.END:
            message = self._interpolate(state.message) if state.message else None
            if message:
                self.context.history.append({"role": "assistant", "content": message})
            logger.info("flow_ended", flow=self.flow.name, slots=self.context.slots)
            return EngineResponse(
                message=message,
                ended=True,
                current_state=self.context.current_state,
                slots=self.context.slots,
            )
        
        if state.type == StateType.ACTION:
            self._pending_action = state.action
            logger.info("action_requested", action=state.action)
            return EngineResponse(
                action_request=state.action,
                current_state=self.context.current_state,
                slots=self.context.slots,
            )
        
        if state.type == StateType.CONDITION:
            next_state = self._evaluate_condition(state)
            if next_state:
                self.context.current_state = next_state
                return self._execute_current_state()
        
        if state.type == StateType.SPEAK:
            message = self._interpolate(state.message) if state.message else None
            if message:
                self.context.history.append({"role": "assistant", "content": message})
            
            if state.next:
                self.context.current_state = state.next
                next_response = self._execute_current_state()
                if message and next_response.message:
                    message = f"{message} {next_response.message}"
                elif next_response.message:
                    message = next_response.message
                return EngineResponse(
                    message=message,
                    needs_input=next_response.needs_input,
                    ended=next_response.ended,
                    action_request=next_response.action_request,
                    current_state=next_response.current_state,
                    slots=self.context.slots,
                )
            
            return EngineResponse(
                message=message,
                current_state=self.context.current_state,
                slots=self.context.slots,
            )
        
        # LISTEN or SPEAK_LISTEN
        message = self._interpolate(state.message) if state.message else None
        if message:
            self.context.history.append({"role": "assistant", "content": message})
        
        return EngineResponse(
            message=message,
            needs_input=True,
            current_state=self.context.current_state,
            slots=self.context.slots,
        )
    
    def _get_current_state(self) -> Optional[FlowState]:
        return self.flow.states.get(self.context.current_state)
    
    def _interpolate(self, text: str) -> str:
        if not text:
            return text
        
        result = text
        
        for key, value in self.context.variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        for key, value in self.context.slots.items():
            result = result.replace(f"{{slots.{key}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    def _evaluate_condition(self, state: FlowState) -> Optional[str]:
        condition = state.condition
        if not condition:
            return state.if_true or state.next
        
        if condition.startswith("slots."):
            parts = condition.split(".")
            if len(parts) >= 2:
                slot_name = parts[1].split()[0]
                if self.context.slots.get(slot_name):
                    return state.if_true
                return state.if_false
        
        return state.if_true or state.next
