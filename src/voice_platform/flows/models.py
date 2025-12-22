"""Flow data models."""
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class StateType(str, Enum):
    """Types of flow states."""
    SPEAK = "speak"                    # Say something, auto-advance
    LISTEN = "listen"                  # Wait for user input, detect intent
    SPEAK_LISTEN = "speak_and_listen"  # Say something, then wait for input
    ACTION = "action"                  # Execute an action (API call, etc.)
    CONDITION = "condition"            # Branch based on context
    END = "end"                        # End the conversation


class Intent(BaseModel):
    """Intent definition for a listen state."""
    patterns: list[str] = Field(default_factory=list)  # Keyword patterns
    examples: list[str] = Field(default_factory=list)  # Example utterances for LLM
    next: str  # Next state if this intent matches


class Slot(BaseModel):
    """Slot to fill from user input."""
    name: str
    type: str = "string"  # string, date, time, phone, number
    prompt: Optional[str] = None  # Re-prompt if not captured
    required: bool = True
    validation: Optional[str] = None  # Regex pattern for validation


class FlowState(BaseModel):
    """A single state in the conversation flow."""
    name: str
    type: StateType
    
    # For SPEAK / SPEAK_LISTEN
    message: Optional[str] = None
    
    # For LISTEN / SPEAK_LISTEN
    intents: dict[str, Intent] = Field(default_factory=dict)
    slots: list[Slot] = Field(default_factory=list)
    fallback_message: str = "I didn't catch that. Could you repeat?"
    max_retries: int = 2
    
    # For ACTION
    action: Optional[str] = None  # e.g., "calendar.check_slots"
    action_params: dict[str, Any] = Field(default_factory=dict)
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    
    # For CONDITION
    condition: Optional[str] = None  # e.g., "slots.appointment_confirmed == true"
    if_true: Optional[str] = None
    if_false: Optional[str] = None
    
    # Default next state (for SPEAK)
    next: Optional[str] = None


class Flow(BaseModel):
    """Complete conversation flow definition."""
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    
    # Initial state
    initial_state: str
    
    # All states
    states: dict[str, FlowState] = Field(default_factory=dict)
    
    # Global fallback
    global_fallback: str = "I'm sorry, something went wrong. Let me transfer you."
    
    # Context variables with defaults
    context_defaults: dict[str, Any] = Field(default_factory=dict)
