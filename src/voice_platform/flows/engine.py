"""Flow execution engine."""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .models import Flow, FlowStep, SlotDefinition, ActionType
from ..core.types import LLMMessage
from ..logging import get_logger

logger = get_logger("flows.engine")


@dataclass
class FlowContext:
    """Execution context for a flow."""
    flow_name: str
    current_step: str
    slots: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    retry_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    
    def set_slot(self, name: str, value: Any) -> None:
        self.slots[name] = value
    
    def get_slot(self, name: str, default: Any = None) -> Any:
        return self.slots.get(name, default)
    
    def get_all(self) -> dict:
        return {**self.variables, **self.slots}
    
    def interpolate(self, text: str) -> str:
        if not text:
            return text
        result = text
        for key, value in self.get_all().items():
            result = result.replace(f"{{{key}}}", str(value) if value else "")
        return result


class SlotExtractor:
    """Extract slot values from user input."""
    
    PATTERNS = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "phone": r"[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}",
        "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        "time": r"\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{1,2}\s*(?:am|pm|AM|PM)|\b(?:morning|afternoon|evening|noon)\b",
        "number": r"\b\d+(?:\.\d+)?\b",
    }
    
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
    
    def extract(self, user_input: str, slot: SlotDefinition, context: FlowContext) -> Optional[Any]:
        text = user_input.strip()
        text_lower = text.lower()
        
        for pattern in slot.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        if slot.type in self.PATTERNS:
            match = re.search(self.PATTERNS[slot.type], text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        if slot.type == "choice" and slot.choices:
            for choice in slot.choices:
                if choice.lower() in text_lower:
                    return choice
        
        if slot.type == "string":
            return text if text else None
        
        if self.llm:
            return self._extract_with_llm(user_input, slot, context)
        
        return None
    
    def _extract_with_llm(self, user_input: str, slot: SlotDefinition, context: FlowContext) -> Optional[str]:
        prompt = f"""Extract the {slot.name} from: "{user_input}"
Type: {slot.type}
{"Choices: " + ", ".join(slot.choices) if slot.choices else ""}
Return ONLY the value or "NONE"."""
        
        messages = [LLMMessage(role="user", content=prompt)]
        response = self.llm.generate(messages, max_tokens=50)
        result = response.content.strip()
        return None if result == "NONE" else result


@dataclass
class FlowResult:
    """Result of a flow step execution."""
    say: Optional[str] = None
    listen: bool = False
    next_step: Optional[str] = None
    end_flow: bool = False
    action: Optional[ActionType] = None
    action_params: dict = field(default_factory=dict)


class FlowEngine:
    """Execute conversation flows."""
    
    def __init__(
        self,
        flows: dict[str, Flow],
        llm: Optional[Any] = None,
        action_handlers: dict[ActionType, Callable] = None,
    ):
        self.flows = flows
        self.extractor = SlotExtractor(llm)
        self.action_handlers = action_handlers or {}
        self.contexts: dict[str, FlowContext] = {}
        logger.info("flow_engine_initialized", flows=list(flows.keys()))
    
    def start_flow(self, session_id: str, flow_name: str) -> FlowResult:
        if flow_name not in self.flows:
            return FlowResult(say="Sorry, I couldn't start that conversation.", end_flow=True)
        
        flow = self.flows[flow_name]
        context = FlowContext(flow_name=flow_name, current_step=flow.initial_step)
        self.contexts[session_id] = context
        
        logger.info("flow_started", session=session_id[:8], flow=flow_name)
        return self._execute_step(session_id)
    
    def process_input(self, session_id: str, user_input: str) -> FlowResult:
        if session_id not in self.contexts:
            return FlowResult(say="No active conversation.", end_flow=True)
        
        context = self.contexts[session_id]
        flow = self.flows[context.flow_name]
        step = flow.get_step(context.current_step)
        
        if not step:
            return FlowResult(say="Something went wrong.", end_flow=True)
        
        # Store input for conditions
        context.variables["last_input"] = user_input.lower()
        context.history.append({
            "step": context.current_step,
            "input": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Extract slots if needed
        for slot in step.extract:
            value = self.extractor.extract(user_input, slot, context)
            if value:
                context.set_slot(slot.name, value)
                logger.debug("slot_extracted", slot=slot.name, value=value)
            elif slot.required:
                context.retry_count += 1
                if context.retry_count <= step.retries:
                    prompt = slot.prompt or f"I didn't catch that. What is your {slot.name}?"
                    return FlowResult(say=context.interpolate(prompt), listen=True)
                else:
                    if slot.default is not None:
                        context.set_slot(slot.name, slot.default)
                    context.retry_count = 0
        
        context.retry_count = 0
        
        # Move to next step then evaluate conditions there
        if step.next_step:
            context.current_step = step.next_step
        
        return self._execute_step(session_id)
    
    def _execute_step(self, session_id: str) -> FlowResult:
        context = self.contexts[session_id]
        flow = self.flows[context.flow_name]
        step = flow.get_step(context.current_step)
        
        if not step:
            return FlowResult(say="Something went wrong.", end_flow=True)
        
        logger.debug("executing_step", step=step.id)
        
        # Handle actions first
        if step.action:
            if step.action == ActionType.END:
                return FlowResult(say=context.interpolate(step.say) if step.say else None, end_flow=True)
            elif step.action == ActionType.HANGUP:
                return FlowResult(action=ActionType.HANGUP, end_flow=True)
            elif step.action == ActionType.TRANSFER:
                return FlowResult(
                    say=context.interpolate(step.say) if step.say else None,
                    action=ActionType.TRANSFER,
                    action_params=step.action_params,
                    end_flow=True,
                )
            elif step.action == ActionType.GOTO:
                target = step.action_params.get("step")
                if target:
                    context.current_step = target
                    return self._execute_step(session_id)
            elif step.action == ActionType.SET:
                for key, value in step.action_params.items():
                    context.variables[key] = context.interpolate(str(value))
            elif step.action in self.action_handlers:
                self.action_handlers[step.action](context, step.action_params)
        
        # If step has conditions, evaluate them
        if step.conditions:
            next_step = self._evaluate_conditions(step, context)
            if next_step:
                context.current_step = next_step
                return self._execute_step(session_id)
            elif step.next_step:
                context.current_step = step.next_step
                return self._execute_step(session_id)
        
        # Build result
        result = FlowResult()
        
        if step.say:
            result.say = context.interpolate(step.say)
        
        result.listen = step.listen
        
        # Auto-advance if not listening and has next step
        if not step.listen and step.next_step:
            context.current_step = step.next_step
            next_result = self._execute_step(session_id)
            if next_result.say:
                result.say = f"{result.say} {next_result.say}" if result.say else next_result.say
            result.listen = next_result.listen
            result.end_flow = next_result.end_flow
            result.action = next_result.action
            result.action_params = next_result.action_params
        
        return result
    
    def _evaluate_conditions(self, step: FlowStep, context: FlowContext) -> Optional[str]:
        all_vars = context.get_all()
        
        for condition in step.conditions:
            if condition.evaluate(all_vars):
                logger.debug(
                    "condition_matched",
                    variable=condition.variable,
                    operator=condition.operator.value,
                    next=condition.next_step,
                )
                return condition.next_step
        
        return None
    
    def get_context(self, session_id: str) -> Optional[FlowContext]:
        return self.contexts.get(session_id)
    
    def end_flow(self, session_id: str) -> None:
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info("flow_ended", session=session_id[:8])
