"""
Tool-Calling Voice AI Agent for Healthcare Scheduling.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import re

from .slot_extractor import SlotExtractor, ExtractedSlots, ConfirmationType
from .llm_slot_extractor import LLMSlotExtractor
from .tools import AppointmentTools, ToolResponse
from ..llm.ollama import OllamaLLM
from ..core.types import LLMMessage
from ..core.config import LLMConfig
from ..logging import get_logger

logger = get_logger("agent.tool_calling")


class AgentState(Enum):
    """Current state of the conversation."""
    GREETING = "greeting"
    COLLECTING_REASON = "collecting_reason"
    COLLECTING_DAY = "collecting_day"
    CONFIRMING_DAY = "confirming_day"  # NEW: Explicit day confirmation
    COLLECTING_TIME = "collecting_time"
    CONFIRMING = "confirming"
    COMPLETE = "complete"
    TRANSFERRED = "transferred"


@dataclass
class ConversationContext:
    """Tracks the full conversation state."""
    state: AgentState = AgentState.GREETING
    
    # Accumulated slots
    visit_reason: Optional[str] = None
    preferred_day: Optional[str] = None
    preferred_time: Optional[str] = None
    
    # Confirmation tracking
    day_confirmed: bool = False
    last_offered_time: Optional[str] = None
    
    # Conversation history
    history: List[Dict[str, str]] = field(default_factory=list)
    
    # Tracking
    turn_count: int = 0
    confirmation_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "visit_reason": self.visit_reason,
            "preferred_day": self.preferred_day,
            "preferred_time": self.preferred_time,
            "turn_count": self.turn_count,
        }
    
    def slots_summary(self) -> str:
        parts = []
        if self.visit_reason:
            parts.append(f"reason: {self.visit_reason}")
        if self.preferred_day:
            parts.append(f"day: {self.preferred_day}")
        if self.preferred_time:
            parts.append(f"time: {self.preferred_time}")
        return ", ".join(parts) if parts else "none collected"
    
    def is_ready_to_book(self) -> bool:
        return all([self.visit_reason, self.preferred_day, self.preferred_time])


@dataclass
class AgentResponse:
    """Response from the agent."""
    text: str
    ended: bool = False
    transfer: bool = False
    booking: Optional[Dict[str, Any]] = None


class ToolCallingAgent:
    """Healthcare appointment scheduling agent."""
    
    SYSTEM_PROMPT = """You are a friendly phone receptionist for {clinic_name}. 

Generate short, natural spoken responses. Keep under 25 words.

CURRENT STATE:
{state_info}

RULES:
- Be warm and conversational
- Keep responses SHORT
- If user mentions pain, acknowledge it briefly

Generate ONLY the spoken response."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        clinic_name: str = "Sunrise Medical",
        available_slots: Optional[List[Dict[str, str]]] = None,
    ):
        self.clinic_name = clinic_name
        self.extractor = SlotExtractor()
        self.llm_extractor = LLMSlotExtractor()
        self.tools = AppointmentTools(clinic_name, available_slots)
        
        if llm is None:
            config = LLMConfig(model="llama3.2:latest", temperature=0.7)
            self.llm = OllamaLLM(config)
        else:
            self.llm = llm
        
        self.context = ConversationContext()
        logger.info("tool_calling_agent_initialized", clinic=clinic_name)

    def start(self) -> str:
        self.context.state = AgentState.COLLECTING_REASON
        greeting = f"Thank you for calling {self.clinic_name}. How can I help you today?"
        self.context.history.append({"role": "assistant", "content": greeting})
        logger.info("agent_started", greeting=greeting[:50])
        return greeting

    def process(self, user_input: str) -> AgentResponse:
        self.context.turn_count += 1
        self.context.history.append({"role": "user", "content": user_input})
        
        # Extract slots
        extracted = self.extractor.extract(user_input)
        
        # Check if user confirmed offered time
        if self.context.last_offered_time and extracted.confirmation == ConfirmationType.YES:
            if not self.context.preferred_time:
                self.context.preferred_time = self.context.last_offered_time
                logger.info("time_confirmed_from_offer", time=self.context.last_offered_time)
        
        # Update context
        self._update_context(extracted)
        
        logger.info(
            "turn_processed",
            turn=self.context.turn_count,
            state=self.context.state.value,
            extracted=extracted.newly_extracted,
            slots=self.context.slots_summary(),
        )
        
        # Handle state
        response = self._handle_state(extracted)
        self.context.history.append({"role": "assistant", "content": response.text})
        return response

    def _update_context(self, extracted: ExtractedSlots) -> None:
        if extracted.visit_reason and not self.context.visit_reason:
            self.context.visit_reason = extracted.visit_reason
        if extracted.preferred_day and not self.context.preferred_day:
            self.context.preferred_day = extracted.preferred_day
        if extracted.preferred_time and not self.context.preferred_time:
            self.context.preferred_time = extracted.preferred_time

    def _handle_state(self, extracted: ExtractedSlots) -> AgentResponse:
        # Check for transfer
        if self._wants_transfer(extracted.raw_text):
            return self._do_transfer("user request")
        
        # Check for goodbye (only if no new slots)
        if self._is_goodbye(extracted.raw_text) and not extracted.newly_extracted:
            if self.context.is_ready_to_book():
                return self._do_booking()
            return AgentResponse(text="Thank you for calling! Have a great day.", ended=True)
        
        # State machine
        if self.context.state == AgentState.COLLECTING_REASON:
            return self._handle_collecting_reason(extracted)
        elif self.context.state == AgentState.COLLECTING_DAY:
            return self._handle_collecting_day(extracted)
        elif self.context.state == AgentState.CONFIRMING_DAY:
            return self._handle_confirming_day(extracted)
        elif self.context.state == AgentState.COLLECTING_TIME:
            return self._handle_collecting_time(extracted)
        elif self.context.state == AgentState.CONFIRMING:
            return self._handle_confirming(extracted)
        else:
            return AgentResponse(text="Thank you for calling. Goodbye!", ended=True)

    def _handle_collecting_reason(self, extracted: ExtractedSlots) -> AgentResponse:
        if self.context.visit_reason:
            self.context.state = AgentState.COLLECTING_DAY
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            
            return AgentResponse(
                text=f"Sorry to hear about your {self.context.visit_reason}. We have openings on {days_str}. Which day works for you?"
            )
        else:
            return AgentResponse(text="What brings you in today?")

    def _handle_collecting_day(self, extracted: ExtractedSlots) -> AgentResponse:
        if self.context.preferred_day:
            availability = self.tools.check_availability(self.context.preferred_day)
            
            if availability.success:
                # Move to day confirmation (Tuesday/Thursday sound similar)
                self.context.state = AgentState.CONFIRMING_DAY
                return AgentResponse(
                    text=f"Just to confirm, you said {self.context.preferred_day}, correct?"
                )
            else:
                self.context.preferred_day = None
                available_days = availability.data.get("available_days", [])
                days_str = ", ".join(available_days[:3])
                return AgentResponse(
                    text=f"Sorry, we're full that day. How about {days_str}?"
                )
        else:
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            return AgentResponse(text=f"What day works for you? We have {days_str} available.")

    def _handle_confirming_day(self, extracted: ExtractedSlots) -> AgentResponse:
        """Handle day confirmation (Tuesday/Thursday disambiguation)."""
        if extracted.confirmation == ConfirmationType.YES:
            self.context.day_confirmed = True
            self.context.state = AgentState.COLLECTING_TIME
            
            availability = self.tools.check_availability(self.context.preferred_day)
            times = availability.data.get("times", [])
            if times:
                self.context.last_offered_time = times[0]
                times_str = " or ".join(times[:2])
                return AgentResponse(
                    text=f"Great! On {self.context.preferred_day} we have {times_str}. Which time works?"
                )
            return AgentResponse(text=f"What time on {self.context.preferred_day}?")
            
        elif extracted.confirmation == ConfirmationType.NO:
            # User said no - they probably meant a different day
            self.context.preferred_day = None
            self.context.state = AgentState.COLLECTING_DAY
            
            # Check if they said the correct day in their response
            if extracted.preferred_day:
                self.context.preferred_day = extracted.preferred_day
                self.context.state = AgentState.CONFIRMING_DAY
                return AgentResponse(
                    text=f"Oh, you meant {extracted.preferred_day}. Is that right?"
                )
            
            return AgentResponse(text="Sorry about that. What day did you mean?")
        else:
            # Unclear - ask again
            return AgentResponse(
                text=f"Sorry, was that {self.context.preferred_day}? Yes or no?"
            )

    def _handle_collecting_time(self, extracted: ExtractedSlots) -> AgentResponse:
        if self.context.is_ready_to_book():
            self.context.state = AgentState.CONFIRMING
            return AgentResponse(
                text=f"Perfect! {self.context.preferred_day} at {self.context.preferred_time} for your {self.context.visit_reason}. Should I book it?"
            )
        
        availability = self.tools.check_availability(self.context.preferred_day)
        available_times = availability.data.get("times", [])
        
        if self.context.preferred_time:
            time_lower = self.context.preferred_time.lower()
            matched_time = None
            for avail_time in available_times:
                if time_lower in avail_time.lower() or avail_time.lower() in time_lower:
                    matched_time = avail_time
                    break
            
            if matched_time:
                self.context.preferred_time = matched_time
                self.context.state = AgentState.CONFIRMING
                return AgentResponse(
                    text=f"Got it! {self.context.preferred_day} at {matched_time}. Should I book this for you?"
                )
            else:
                self.context.preferred_time = None
                self.context.last_offered_time = available_times[0] if available_times else None
                times_str = " or ".join(available_times[:2])
                return AgentResponse(
                    text=f"That time isn't available. We have {times_str}. Which works?"
                )
        else:
            # No time extracted by regex - try LLM for implicit preferences
            if available_times:
                # Use LLM to understand implicit preferences
                llm_result = self.llm_extractor.extract(
                    user_text=extracted.raw_text,
                    collecting="time",
                    available_days=[],
                    available_times=available_times,
                )
                
                if llm_result.time_preference:
                    # User expressed a preference like "I work in the morning"
                    logger.info("llm_time_preference", preference=llm_result.time_preference)
                    
                    # Filter times based on preference
                    if llm_result.time_preference == "afternoon":
                        afternoon_times = [t for t in available_times if "PM" in t]
                        if afternoon_times:
                            self.context.preferred_time = afternoon_times[0]
                            self.context.state = AgentState.CONFIRMING
                            return AgentResponse(
                                text=f"Since you're busy in the morning, how about {afternoon_times[0]}?"
                            )
                    elif llm_result.time_preference == "morning":
                        morning_times = [t for t in available_times if "AM" in t]
                        if morning_times:
                            self.context.preferred_time = morning_times[0]
                            self.context.state = AgentState.CONFIRMING
                            return AgentResponse(
                                text=f"Since you prefer morning, how about {morning_times[0]}?"
                            )
                
                # No preference inferred, just ask
                self.context.last_offered_time = available_times[0]
                times_str = " or ".join(available_times[:2])
                return AgentResponse(text=f"We have {times_str}. Which time works for you?")
            return AgentResponse(text=f"{self.context.preferred_day} is full. Try another day?")

    def _handle_confirming(self, extracted: ExtractedSlots) -> AgentResponse:
        if extracted.confirmation == ConfirmationType.YES:
            return self._do_booking()
        elif extracted.confirmation == ConfirmationType.NO:
            self.context.state = AgentState.COLLECTING_REASON
            self.context.visit_reason = None
            self.context.preferred_day = None
            self.context.preferred_time = None
            self.context.day_confirmed = False
            return AgentResponse(text="No problem. What would you like to change?")
        else:
            self.context.confirmation_attempts += 1
            if self.context.confirmation_attempts >= 2:
                return self._do_transfer("unclear confirmation")
            return AgentResponse(
                text=f"{self.context.preferred_day} at {self.context.preferred_time}. Book it? Yes or no?"
            )

    def _do_booking(self) -> AgentResponse:
        result = self.tools.book_appointment(
            reason=self.context.visit_reason,
            day=self.context.preferred_day,
            time=self.context.preferred_time,
        )
        
        if result.success:
            self.context.state = AgentState.COMPLETE
            conf_num = result.data.get("confirmation_number", "")
            return AgentResponse(
                text=f"You're all set for {self.context.preferred_day} at {self.context.preferred_time}. Confirmation number {conf_num}. See you then!",
                ended=True,
                booking=result.data,
            )
        return AgentResponse(text=f"Problem booking: {result.message}. Try a different time?")

    def _generate_response(self, instruction: str) -> str:
        state_info = f"State: {self.context.state.value}\nCollected: {self.context.slots_summary()}\nInstruction: {instruction}"
        system_prompt = self.SYSTEM_PROMPT.format(clinic_name=self.clinic_name, state_info=state_info)
        messages = [LLMMessage(role="system", content=system_prompt)]
        for turn in self.context.history[-4:]:
            messages.append(LLMMessage(role=turn["role"], content=turn["content"]))
        
        try:
            response = self.llm.generate(messages, max_tokens=60)
            return response.content.strip().strip('"').strip("'")
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return "How can I help you?"

    def _wants_transfer(self, text: str) -> bool:
        phrases = ["speak to someone", "talk to someone", "human", "person", "representative", "transfer", "operator"]
        return any(p in text.lower() for p in phrases)

    def _is_goodbye(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if any(text_lower == g or text_lower.startswith(g + " ") or text_lower.endswith(" " + g) 
               for g in ["bye", "goodbye", "good bye", "see you"]):
            return True
        if "that's all" in text_lower or "nothing else" in text_lower:
            return True
        if re.search(r"i'?m good\.?$", text_lower):
            return True
        return False

    def _do_transfer(self, reason: str) -> AgentResponse:
        self.context.state = AgentState.TRANSFERRED
        self.tools.transfer_to_human(reason)
        return AgentResponse(text="I'll transfer you to a staff member. Please hold.", ended=True, transfer=True)

    def get_context(self) -> Dict[str, Any]:
        return self.context.to_dict()
