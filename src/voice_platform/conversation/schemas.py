"""Conversation schemas."""
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field


class Stage(str, Enum):
    """Simple conversation stages."""
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    BOOKING = "booking"
    DONE = "done"


@dataclass
class ConversationState:
    """Current state of the conversation."""
    stage: Stage = Stage.GREETING
    slots: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    confirmed: bool = False
    
    def get_missing_slots(self, required: list[str]) -> list[str]:
        return [s for s in required if not self.slots.get(s)]
    
    def to_context(self) -> str:
        parts = [f"Stage: {self.stage.value}"]
        if self.slots:
            parts.append(f"Collected: {self.slots}")
        missing = self.get_missing_slots(["visit_reason", "preferred_day", "preferred_time"])
        if missing:
            parts.append(f"Still need: {missing}")
        return " | ".join(parts)


@dataclass  
class AgentResponse:
    """Response from the conversation agent."""
    message: str
    slots: dict[str, Any]
    stage: Stage
    ready_to_book: bool = False
    ended: bool = False
