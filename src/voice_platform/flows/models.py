"""Flow data models."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionType(Enum):
    """Types of actions a step can perform."""
    SAY = "say"              # Speak text
    LISTEN = "listen"        # Wait for user input
    EXTRACT = "extract"      # Extract slot from input
    SET = "set"              # Set a variable
    CALL_API = "call_api"    # Call external API
    TRANSFER = "transfer"    # Transfer call (telephony)
    HANGUP = "hangup"        # End call
    GOTO = "goto"            # Jump to step
    END = "end"              # End flow


class ConditionOperator(Enum):
    """Condition comparison operators."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    MATCHES = "matches"      # Regex match
    IN = "in"                # Value in list


@dataclass
class FlowCondition:
    """Condition for branching logic."""
    variable: str
    operator: ConditionOperator
    value: Any
    next_step: str
    
    def evaluate(self, context: dict) -> bool:
        """Evaluate condition against context."""
        var_value = context.get(self.variable)
        
        if self.operator == ConditionOperator.EQUALS:
            return var_value == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return var_value != self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in str(var_value) if var_value else False
        elif self.operator == ConditionOperator.NOT_CONTAINS:
            return self.value not in str(var_value) if var_value else True
        elif self.operator == ConditionOperator.GREATER_THAN:
            return float(var_value) > float(self.value) if var_value else False
        elif self.operator == ConditionOperator.LESS_THAN:
            return float(var_value) < float(self.value) if var_value else False
        elif self.operator == ConditionOperator.EXISTS:
            return var_value is not None
        elif self.operator == ConditionOperator.NOT_EXISTS:
            return var_value is None
        elif self.operator == ConditionOperator.IN:
            return var_value in self.value if isinstance(self.value, list) else False
        elif self.operator == ConditionOperator.MATCHES:
            import re
            return bool(re.match(self.value, str(var_value))) if var_value else False
        
        return False


@dataclass
class SlotDefinition:
    """Definition for extracting slots from user input."""
    name: str
    type: str = "string"          # string, number, date, time, phone, email, choice
    prompt: Optional[str] = None  # Re-prompt if extraction fails
    required: bool = True
    choices: list[str] = field(default_factory=list)  # For choice type
    patterns: list[str] = field(default_factory=list)  # Regex patterns
    default: Any = None


@dataclass
class FlowStep:
    """A single step in a conversation flow."""
    id: str
    say: Optional[str] = None           # Text to speak
    listen: bool = False                 # Wait for user input
    extract: list[SlotDefinition] = field(default_factory=list)
    conditions: list[FlowCondition] = field(default_factory=list)
    next_step: Optional[str] = None     # Default next step
    action: Optional[ActionType] = None
    action_params: dict = field(default_factory=dict)
    retries: int = 2                    # Retries for slot extraction
    timeout_seconds: int = 10
    on_timeout: Optional[str] = None    # Step to go on timeout
    on_error: Optional[str] = None      # Step to go on error


@dataclass  
class Flow:
    """Complete conversation flow definition."""
    name: str
    description: str = ""
    version: str = "1.0"
    initial_step: str = "start"
    steps: dict[str, FlowStep] = field(default_factory=dict)
    global_slots: list[SlotDefinition] = field(default_factory=list)
    fallback_step: Optional[str] = None  # When no condition matches
    metadata: dict = field(default_factory=dict)
    
    def get_step(self, step_id: str) -> Optional[FlowStep]:
        """Get step by ID."""
        return self.steps.get(step_id)
    
    def validate(self) -> list[str]:
        """Validate flow structure, return list of errors."""
        errors = []
        
        # Check initial step exists
        if self.initial_step not in self.steps:
            errors.append(f"Initial step '{self.initial_step}' not found")
        
        # Check all referenced steps exist
        for step_id, step in self.steps.items():
            if step.next_step and step.next_step not in self.steps:
                errors.append(f"Step '{step_id}' references unknown step '{step.next_step}'")
            
            for cond in step.conditions:
                if cond.next_step not in self.steps:
                    errors.append(f"Step '{step_id}' condition references unknown step '{cond.next_step}'")
            
            if step.on_timeout and step.on_timeout not in self.steps:
                errors.append(f"Step '{step_id}' on_timeout references unknown step '{step.on_timeout}'")
            
            if step.on_error and step.on_error not in self.steps:
                errors.append(f"Step '{step_id}' on_error references unknown step '{step.on_error}'")
        
        return errors
