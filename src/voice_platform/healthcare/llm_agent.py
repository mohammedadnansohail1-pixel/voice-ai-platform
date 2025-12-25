"""Healthcare appointment agent with State Machine dialog management."""
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..llm import OllamaLLM
from ..core.types import LLMMessage
from ..logging import get_logger
from ..conversation.guardrails import ConversationGuardrails, GuardrailAction

logger = get_logger("healthcare.llm_agent")


class DialogState(Enum):
    """State machine states for appointment booking."""
    GREETING = "greeting"
    ASK_REASON = "ask_reason"
    ASK_DAY = "ask_day"
    ASK_TIME = "ask_time"
    CONFIRM = "confirm"
    COMPLETE = "complete"
    TRANSFER = "transfer"


@dataclass 
class AppointmentContext:
    """Tracks extracted appointment information."""
    visit_reason: Optional[str] = None
    preferred_day: Optional[str] = None
    preferred_time: Optional[str] = None


class HealthcareLLMAgent:
    """
    Healthcare scheduling agent using State Machine + LLM.
    
    State Machine controls dialog flow (what to ask).
    Rule-based extraction validates slots.
    LLM generates natural responses.
    """
    
    # Slot extraction patterns
    DAY_PATTERNS = {
        "monday": ["monday", "mon"],
        "tuesday": ["tuesday", "tue", "tues"],
        "wednesday": ["wednesday", "wed"],
        "thursday": ["thursday", "thu", "thur", "thurs"],
        "friday": ["friday", "fri"],
        "saturday": ["saturday", "sat"],
        "sunday": ["sunday", "sun"],
    }
    
    TIME_PATTERNS = [
        (r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm|a\.m\.|p\.m\.)", lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3).upper().replace('.', '')}"),
        (r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", lambda m: f"{m.group(1)}:00 {m.group(2).upper().replace('.', '')}"),
        (r"\b(10|ten)\b.*\b(morning|am)\b", lambda m: "10:00 AM"),
        (r"\bmorning\b", lambda m: "10:00 AM"),
        (r"\b(2|two)\b.*\b(afternoon|pm)\b", lambda m: "2:00 PM"),
        (r"\bafternoon\b", lambda m: "2:00 PM"),
        (r"\b(3|three)\b.*\b(pm|afternoon)\b", lambda m: "3:00 PM"),
        (r"\b(4|four)\b.*\b(pm|afternoon)\b", lambda m: "4:00 PM"),
    ]
    
    REASON_PATTERNS = [
        r"(toothache|tooth\s*ache|tooth\s*pain)",
        r"(headache|head\s*ache|head\s*pain)",
        r"(backache|back\s*ache|back\s*pain)",
        r"(checkup|check-up|check\s*up|physical|annual)",
        r"(cleaning|teeth\s*cleaning)",
        r"(pain|ache|hurt|hurting)",
        r"(cavity|filling|crown|root\s*canal)",
        r"(fever|cough|cold|flu|sick)",
    ]

    def __init__(
        self,
        llm: OllamaLLM,
        clinic_name: str = "Sunrise Medical",
        available_slots: Optional[List[tuple]] = None,
    ):
        self.llm = llm
        self.clinic_name = clinic_name
        self.available_slots = available_slots or [
            ("Tuesday", "2:00 PM"),
            ("Tuesday", "4:30 PM"),
            ("Wednesday", "10:00 AM"),
            ("Wednesday", "3:00 PM"),
            ("Thursday", "9:00 AM"),
            ("Friday", "11:00 AM"),
        ]
        
        self.context = AppointmentContext()
        self.state = DialogState.GREETING
        self.ended = False
        self.transfer_requested = False
        self.turn_count = 0
        
        # Initialize guardrails
        self.guardrails = ConversationGuardrails(clinic_name=clinic_name)
        
        logger.info("healthcare_llm_agent_init", clinic=clinic_name, state=self.state.value)

    def _extract_day(self, text: str) -> Optional[str]:
        """Extract day from text using patterns."""
        text_lower = text.lower()
        for day, variants in self.DAY_PATTERNS.items():
            if any(v in text_lower for v in variants):
                return day.capitalize()
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time from text using patterns."""
        text_lower = text.lower()
        for pattern, formatter in self.TIME_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return formatter(match)
        return None

    def _extract_reason(self, text: str) -> Optional[str]:
        """Extract visit reason from text."""
        text_lower = text.lower()
        for pattern in self.REASON_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        return None

    def _get_available_days(self) -> str:
        """Get unique available days."""
        days = sorted(set(day for day, _ in self.available_slots))
        return ", ".join(days)

    def _get_times_for_day(self, day: str) -> List[str]:
        """Get available times for a specific day."""
        return [time for d, time in self.available_slots if d.lower() == day.lower()]

    def _format_slots(self) -> str:
        """Format available slots."""
        return ", ".join([f"{day} at {time}" for day, time in self.available_slots])

    def _generate_response(self, template: str) -> str:
        """Use LLM to make template response more natural."""
        messages = [
            LLMMessage(role="system", content=f"You are a friendly phone receptionist for {self.clinic_name}. Rephrase the following in a natural, warm way. Keep it under 20 words. Output ONLY the rephrased text, nothing else."),
            LLMMessage(role="user", content=template)
        ]
        
        try:
            response = self.llm.generate(messages, max_tokens=50)
            return response.content.strip().strip('"')
        except Exception as e:
            logger.warning("llm_rephrase_failed", error=str(e))
            return template

    def start(self) -> str:
        """Start conversation with greeting."""
        self.state = DialogState.ASK_REASON
        return f"Thank you for calling {self.clinic_name}. What brings you in today?"

    def process(self, user_input: str) -> str:
        """Process user input through state machine."""
        self.turn_count += 1
        
        # Check guardrails FIRST
        guardrail_result = self.guardrails.check(user_input, self._get_context_dict())
        
        if guardrail_result.action == GuardrailAction.EMERGENCY:
            self.state = DialogState.TRANSFER
            self.ended = True
            return guardrail_result.response
            
        if guardrail_result.action == GuardrailAction.TRANSFER:
            self.state = DialogState.TRANSFER
            self.ended = True
            self.transfer_requested = True
            return guardrail_result.response
        
        # Extract all possible slots from input
        extracted_day = self._extract_day(user_input)
        extracted_time = self._extract_time(user_input)
        extracted_reason = self._extract_reason(user_input)
        
        # Update context with valid extractions
        if extracted_reason and not self.context.visit_reason:
            self.context.visit_reason = extracted_reason
            logger.info("slot_filled", slot="reason", value=extracted_reason)
            
        if extracted_day and not self.context.preferred_day:
            self.context.preferred_day = extracted_day
            logger.info("slot_filled", slot="day", value=extracted_day)
            
        if extracted_time and not self.context.preferred_time:
            self.context.preferred_time = extracted_time
            logger.info("slot_filled", slot="time", value=extracted_time)
        
        # State machine transitions
        response = self._handle_state(user_input)
        
        logger.info("state_machine",
            state=self.state.value,
            reason=self.context.visit_reason,
            day=self.context.preferred_day,
            time=self.context.preferred_time,
            response=response[:60] if response else None,
        )
        
        return response

    def _handle_state(self, user_input: str) -> str:
        """Handle current state and transition."""
        
        # ASK_REASON state
        if self.state == DialogState.ASK_REASON:
            if self.context.visit_reason:
                # Got reason, move to day
                self.state = DialogState.ASK_DAY
                days = self._get_available_days()
                return f"Sorry to hear that. We have openings on {days}. What day works for you?"
            else:
                # Still need reason
                return "Could you tell me what brings you in today? For example, a toothache, checkup, or cleaning?"
        
        # ASK_DAY state
        elif self.state == DialogState.ASK_DAY:
            if self.context.preferred_day:
                # Got day, check if we have slots
                times = self._get_times_for_day(self.context.preferred_day)
                if times:
                    self.state = DialogState.ASK_TIME
                    times_str = " or ".join(times)
                    return f"Great! On {self.context.preferred_day}, we have {times_str}. What time works for you?"
                else:
                    # No slots on that day
                    self.context.preferred_day = None
                    days = self._get_available_days()
                    return f"Sorry, we don't have openings on that day. We have {days} available. Which works for you?"
            else:
                days = self._get_available_days()
                return f"What day works best for you? We have {days} available."
        
        # ASK_TIME state
        elif self.state == DialogState.ASK_TIME:
            if self.context.preferred_time:
                # Validate time is available
                times = self._get_times_for_day(self.context.preferred_day)
                if self.context.preferred_time in times:
                    self.state = DialogState.CONFIRM
                    return f"Perfect! So that's {self.context.preferred_day} at {self.context.preferred_time} for {self.context.visit_reason}. Should I book that for you?"
                else:
                    # Time not available
                    self.context.preferred_time = None
                    times_str = " or ".join(times)
                    return f"Sorry, that time isn't available. We have {times_str} on {self.context.preferred_day}. Which would you prefer?"
            else:
                times = self._get_times_for_day(self.context.preferred_day)
                times_str = " or ".join(times)
                return f"What time works for you on {self.context.preferred_day}? We have {times_str}."
        
        # CONFIRM state
        elif self.state == DialogState.CONFIRM:
            input_lower = user_input.lower()
            if any(w in input_lower for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "book", "confirm"]):
                self.state = DialogState.COMPLETE
                self.ended = True
                return f"Your appointment is booked for {self.context.preferred_day} at {self.context.preferred_time}. We'll see you then. Goodbye!"
            elif any(w in input_lower for w in ["no", "nope", "change", "different", "cancel", "wait"]):
                self.state = DialogState.ASK_REASON
                self.context = AppointmentContext()  # Reset
                return "No problem. Let's start over. What brings you in today?"
            else:
                return f"Just to confirm: {self.context.preferred_day} at {self.context.preferred_time} for {self.context.visit_reason}. Is that correct?"
        
        # COMPLETE or TRANSFER - shouldn't reach here
        else:
            return "Thank you for calling. Goodbye!"

    def _get_context_dict(self) -> Dict[str, Any]:
        """Get context as dictionary."""
        return {
            "visit_reason": self.context.visit_reason,
            "preferred_day": self.context.preferred_day,
            "preferred_time": self.context.preferred_time,
        }

    def get_context(self) -> Dict[str, Any]:
        """Get current context including state."""
        return {
            "visit_reason": self.context.visit_reason,
            "preferred_day": self.context.preferred_day,
            "preferred_time": self.context.preferred_time,
            "state": self.state.value,
            "ended": self.ended,
            "transfer_requested": self.transfer_requested,
        }
