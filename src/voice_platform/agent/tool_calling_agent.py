"""
Tool-Calling Voice AI Agent for Healthcare Scheduling.

Collects patient information with HIPAA compliance:
- Verbal consent before data collection
- Name, DOB, Phone collection
- Encrypted storage via secure-core
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import re
import uuid

from .extractors.slot_extractor import SlotExtractor, ExtractedSlots, ConfirmationType
from .extractors.llm_slot_extractor import LLMSlotExtractor
from .tools.appointment import AppointmentTools, ToolResponse
from ..llm.ollama import OllamaLLM
from ..core.types import LLMMessage
from ..core.config import LLMConfig
from ..logging import get_logger

logger = get_logger("agent.tool_calling")


class AgentState(Enum):
    """Current state of the conversation."""
    GREETING = "greeting"
    # Patient info collection (NEW)
    COLLECTING_CONSENT = "collecting_consent"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_DOB = "collecting_dob"
    COLLECTING_PHONE = "collecting_phone"
    # Appointment booking
    COLLECTING_REASON = "collecting_reason"
    COLLECTING_DAY = "collecting_day"
    CONFIRMING_DAY = "confirming_day"
    COLLECTING_TIME = "collecting_time"
    CONFIRMING = "confirming"
    COMPLETE = "complete"
    TRANSFERRED = "transferred"


@dataclass
class PatientInfo:
    """Collected patient information."""
    consent_given: bool = False
    consent_method: str = "verbal"
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    phone: Optional[str] = None
    patient_id: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if minimum required info is collected."""
        return self.consent_given and self.full_name and self.phone


@dataclass 
class ConversationContext:
    """Tracks the full conversation state."""
    state: AgentState = AgentState.GREETING
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Patient info (NEW)
    patient: PatientInfo = field(default_factory=PatientInfo)
    
    # Appointment slots
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

    def is_ready_to_book(self) -> bool:
        return all([
            self.visit_reason,
            self.preferred_day,
            self.preferred_time,
            self.day_confirmed,
            self.patient.is_complete(),
        ])


@dataclass
class AgentResponse:
    """Response from agent processing."""
    text: str
    state: Optional[AgentState] = None
    ended: bool = False
    booking: Optional[Dict[str, Any]] = None
    patient_info: Optional[Dict[str, Any]] = None


class ToolCallingAgent:
    """
    Voice AI agent for healthcare appointment scheduling.
    
    Flow:
    1. Greeting
    2. Consent collection (HIPAA required)
    3. Patient info (name, DOB, phone)
    4. Appointment booking (reason, day, time)
    5. Confirmation
    """
    
    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        clinic_name: str = "Sunrise Medical",
        available_slots: Optional[Dict[str, List[str]]] = None,
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
        """Start conversation with greeting."""
        self.context.state = AgentState.COLLECTING_CONSENT
        greeting = (
            f"Thank you for calling {self.clinic_name}. "
            "Before we proceed, I need to let you know this call may be recorded for quality purposes, "
            "and I'll need to collect some personal information to schedule your appointment. "
            "Do I have your consent to continue?"
        )
        self.context.history.append({"role": "assistant", "content": greeting})
        logger.info("agent_started", greeting=greeting[:50])
        return greeting

    def process(self, user_input: str) -> AgentResponse:
        """Process user input and return response."""
        self.context.turn_count += 1
        self.context.history.append({"role": "user", "content": user_input})
        
        # Extract slots
        extracted = self.extractor.extract(user_input)
        
        # Log extraction
        slots_summary = self._get_slots_summary()
        logger.info(
            "turn_processed",
            turn=self.context.turn_count,
            state=self.context.state.value,
            extracted=extracted.newly_extracted,
            slots=slots_summary,
        )
        
        # Handle based on state
        response = self._handle_state(user_input, extracted)
        
        self.context.history.append({"role": "assistant", "content": response.text})
        response.state = self.context.state
        
        return response

    def _handle_state(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Route to appropriate handler based on state."""
        handlers = {
            AgentState.COLLECTING_CONSENT: self._handle_consent,
            AgentState.COLLECTING_NAME: self._handle_name,
            AgentState.COLLECTING_DOB: self._handle_dob,
            AgentState.COLLECTING_PHONE: self._handle_phone,
            AgentState.COLLECTING_REASON: self._handle_collecting_reason,
            AgentState.COLLECTING_DAY: self._handle_collecting_day,
            AgentState.CONFIRMING_DAY: self._handle_confirming_day,
            AgentState.COLLECTING_TIME: self._handle_collecting_time,
            AgentState.CONFIRMING: self._handle_confirming,
        }
        
        handler = handlers.get(self.context.state)
        if handler:
            return handler(user_input, extracted)
        
        return AgentResponse(text="How can I help you?")

    # === PATIENT INFO HANDLERS ===
    
    def _check_correction_intent(self, user_input: str) -> Optional[AgentResponse]:
        """Check for correction intent globally - user wants to fix something."""
        lower_input = user_input.lower()
        
        # Patterns indicating user wants to correct something
        correction_patterns = [
            r"(name|that).*(wrong|incorrect|not right|mistake)",
            r"(wrong|incorrect|not right).*(name|that)",
            r"you (got|have) .*(wrong|incorrect)",
            r"that's not (my|correct|right)",
            r"i (didn't|did not) say",
            r"(change|correct|fix|update) (my|the)",
            r"let me (correct|fix|change)",
            r"go back",
        ]
        
        for pattern in correction_patterns:
            if re.search(pattern, lower_input):
                logger.info("correction_intent_detected", pattern=pattern)
                return self._handle_correction_request(lower_input)
        
        return None
    
    def _handle_correction_request(self, user_input: str) -> AgentResponse:
        """Handle user wanting to correct information."""
        lower_input = user_input.lower()
        
        # Detect what they want to correct
        if "name" in lower_input:
            self.context.state = AgentState.COLLECTING_NAME
            self.context.patient.full_name = None
            return AgentResponse(text="No problem. What is your correct name?")
        elif "birth" in lower_input or "dob" in lower_input or "date" in lower_input:
            self.context.state = AgentState.COLLECTING_DOB
            self.context.patient.date_of_birth = None
            return AgentResponse(text="No problem. What is your correct date of birth?")
        elif "phone" in lower_input or "number" in lower_input:
            self.context.state = AgentState.COLLECTING_PHONE
            self.context.patient.phone = None
            return AgentResponse(text="No problem. What is your correct phone number?")
        elif "day" in lower_input or "date" in lower_input:
            self.context.state = AgentState.COLLECTING_DAY
            self.context.preferred_day = None
            return AgentResponse(text="No problem. What day would you prefer?")
        elif "time" in lower_input:
            self.context.state = AgentState.COLLECTING_TIME
            self.context.preferred_time = None
            return AgentResponse(text="No problem. What time works for you?")
        else:
            # Generic - ask what to change
            return AgentResponse(
                text="I understand you want to make a correction. "
                     "What would you like to change: your name, date of birth, phone, day, or time?"
            )

    def _handle_consent(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle consent collection."""
        input_lower = user_input.lower()
        
        # Check for consent
        if any(word in input_lower for word in ["yes", "yeah", "sure", "okay", "ok", "consent", "agree"]):
            self.context.patient.consent_given = True
            self.context.patient.consent_method = "verbal"
            self.context.state = AgentState.COLLECTING_NAME
            
            logger.info("consent_collected", method="verbal", session=self.context.session_id)
            
            return AgentResponse(
                text="Thank you. May I have your full name please?"
            )
        elif any(word in input_lower for word in ["no", "don't", "refuse", "disagree"]):
            return AgentResponse(
                text="I understand. Without your consent, I won't be able to schedule an appointment. "
                     "Is there anything else I can help you with today?",
                ended=True,
            )
        else:
            return AgentResponse(
                text="I need your verbal consent to collect your information and schedule an appointment. "
                     "Do I have your consent to continue? Please say yes or no."
            )

    def _handle_name(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle name collection."""
        # Extract name - simple approach: use the input as name if it looks like a name
        name = self._extract_name(user_input)
        
        if name:
            # Heuristic: single very short name is likely ASR error
            if len(name) < 3 or (len(name.split()) == 1 and len(name) < 4):
                logger.info("name_too_short", name=name)
                return AgentResponse(
                    text=f"I heard '{name}'. Could you say your full name, first and last?"
                )
            
            self.context.patient.full_name = name
            self.context.state = AgentState.COLLECTING_DOB
            
            logger.info("name_collected", name_masked=name[:3] + "***")
            
            return AgentResponse(
                text=f"Thanks {name.split()[0]}. If I got that wrong, just say so. What's your date of birth?"
            )
        else:
            return AgentResponse(
                text="I didn't catch that. Could you please tell me your full name?"
            )

    def _handle_dob(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle date of birth collection."""
        dob = self._extract_date(user_input)
        
        if dob:
            self.context.patient.date_of_birth = dob
            self.context.state = AgentState.COLLECTING_PHONE
            
            logger.info("dob_collected", year=dob.split("/")[-1] if "/" in dob else "***")
            
            return AgentResponse(
                text=f"Got it, {dob}. What's the best phone number to reach you?"
            )
        else:
            # Allow skip for DOB
            if any(word in user_input.lower() for word in ["skip", "prefer not", "don't want"]):
                self.context.state = AgentState.COLLECTING_PHONE
                return AgentResponse(
                    text="No problem. What's the best phone number to reach you?"
                )
            
            return AgentResponse(
                text="I need your date of birth for verification. "
                     "Please say it like: March 15, 1985. Or say 'skip' to continue."
            )

    def _handle_phone(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle phone collection."""
        phone = self._extract_phone(user_input)
        
        if phone:
            self.context.patient.phone = phone
            self.context.patient.patient_id = str(uuid.uuid4())
            self.context.state = AgentState.COLLECTING_REASON
            
            # Mask for logging
            masked_phone = phone[:3] + "-***-" + phone[-4:] if len(phone) >= 10 else "***"
            logger.info("phone_collected", phone_masked=masked_phone)
            logger.info(
                "patient_info_complete",
                patient_id=self.context.patient.patient_id[:8],
                session_id=self.context.session_id[:8],
            )
            
            return AgentResponse(
                text="Thank you for that information. Now, what brings you in today?"
            )
        else:
            return AgentResponse(
                text="I need a phone number to confirm your appointment. "
                     "Please say your 10-digit phone number."
            )

    # === EXTRACTION HELPERS ===
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from text with production heuristics."""
        # Remove common prefixes
        text = re.sub(r"^(my name is|i'm|i am|this is|it's|name is|call me)\s*", "", text.lower()).strip()
        
        # Remove punctuation
        text = re.sub(r"[.,!?]", "", text)
        
        # Get words that look like name parts
        words = text.split()
        name_words = []
        
        # Common non-name words (ASR artifacts, filler words)
        skip_words = {"the", "is", "my", "its", "and", "to", "a", "an", "for", "of", "um", "uh"}
        
        for word in words[:4]:  # Take up to 4 words
            # Skip very short words (except valid short names like "Al", "Bo", "Jo")
            if len(word) < 2:
                continue
            if word in skip_words:
                continue
            # Accept if mostly letters
            if sum(c.isalpha() for c in word) >= len(word) * 0.7:
                name_words.append(word.capitalize())
        
        if name_words:
            return " ".join(name_words)
        
        # Fallback: just use first 3 words if they have letters
        fallback_words = [w.capitalize() for w in words[:3] if any(c.isalpha() for c in w)]
        if fallback_words:
            return " ".join(fallback_words)
        
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text - lenient for ASR errors."""
        text_lower = text.lower()
        
        # Month names (including common ASR errors)
        month_map = {
            "january": "01", "jan": "01",
            "february": "02", "feb": "02",
            "march": "03", "mar": "03",
            "april": "04", "apr": "04",
            "may": "05",
            "june": "06", "jun": "06",
            "july": "07", "jul": "07",
            "august": "08", "aug": "08",
            "september": "09", "sep": "09", "sept": "09",
            "october": "10", "oct": "10",
            "november": "11", "nov": "11",
            "december": "12", "dec": "12",
        }
        
        # Try standard patterns first
        patterns = [
            r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",  # MM/DD/YYYY
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[0] in month_map:
                        month = month_map[groups[0]]
                        day = groups[1].zfill(2)
                        year = groups[2]
                    else:
                        month = groups[0].zfill(2)
                        day = groups[1].zfill(2)
                        year = groups[2] if len(groups[2]) == 4 else f"19{groups[2]}"
                    return f"{month}/{day}/{year}"
        
        # Lenient: look for any 4-digit year (1900-2025)
        year_match = re.search(r"(19\d{2}|20[0-2]\d)", text)
        if year_match:
            year = year_match.group(1)
            
            # Try to find month name
            month = "01"
            for m_name, m_num in month_map.items():
                if m_name in text_lower:
                    month = m_num
                    break
            
            # Try to find day (1-31)
            day_match = re.search(r"\b([1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\b", text_lower)
            day = day_match.group(1).zfill(2) if day_match else "15"
            
            return f"{month}/{day}/{year}"
        
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text."""
        # Remove non-digits
        digits = re.sub(r"\D", "", text)
        
        # US phone: 10 digits, or 11 with leading 1
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        return None

    # === APPOINTMENT HANDLERS (existing logic) ===
    
    def _handle_collecting_reason(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle visit reason collection."""
        if extracted.visit_reason:
            self.context.visit_reason = extracted.visit_reason
        
        if self.context.visit_reason:
            self.context.state = AgentState.COLLECTING_DAY
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            return AgentResponse(
                text=f"Sorry to hear about your {self.context.visit_reason}. "
                     f"We have openings on {days_str}. Which day works for you?"
            )
        else:
            return AgentResponse(text="What brings you in today?")

    def _handle_collecting_day(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle day collection."""
        if extracted.preferred_day:
            self.context.preferred_day = extracted.preferred_day
        
        if self.context.preferred_day:
            availability = self.tools.check_availability(self.context.preferred_day)
            if availability.success:
                self.context.state = AgentState.CONFIRMING_DAY
                return AgentResponse(
                    text=f"Just to confirm, you said {self.context.preferred_day}, correct?"
                )
            else:
                self.context.preferred_day = None
                availability = self.tools.check_availability()
                days = list(availability.data.get("days", []))
                days_str = ", ".join(days[:3])
                return AgentResponse(
                    text=f"Sorry, that day isn't available. We have {days_str}. Which works for you?"
                )
        else:
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            return AgentResponse(text=f"What day works for you? We have {days_str} available.")

    def _handle_confirming_day(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle day confirmation."""
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
            self.context.preferred_day = None
            self.context.state = AgentState.COLLECTING_DAY
            if extracted.preferred_day:
                self.context.preferred_day = extracted.preferred_day
                self.context.state = AgentState.CONFIRMING_DAY
                return AgentResponse(
                    text=f"Oh, you meant {extracted.preferred_day}. Is that right?"
                )
            return AgentResponse(text="Sorry about that. What day did you mean?")
        else:
            return AgentResponse(
                text=f"Sorry, was that {self.context.preferred_day}? Yes or no?"
            )

    def _handle_collecting_time(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle time collection with LLM preference understanding."""
        if extracted.preferred_time:
            self.context.preferred_time = extracted.preferred_time
        
        if self.context.is_ready_to_book():
            self.context.state = AgentState.CONFIRMING
            return AgentResponse(
                text=f"Perfect! {self.context.preferred_day} at {self.context.preferred_time} "
                     f"for your {self.context.visit_reason}. Should I book it?"
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
                llm_result = self.llm_extractor.extract(
                    user_text=extracted.raw_text,
                    collecting="time",
                    available_days=[],
                    available_times=available_times,
                )
                
                if llm_result.time_preference:
                    logger.info("llm_time_preference", preference=llm_result.time_preference)
                    
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
                
                self.context.last_offered_time = available_times[0]
                times_str = " or ".join(available_times[:2])
                return AgentResponse(text=f"We have {times_str}. Which time works for you?")
            return AgentResponse(text=f"{self.context.preferred_day} is full. Try another day?")

    def _handle_confirming(self, user_input: str, extracted: ExtractedSlots) -> AgentResponse:
        """Handle final confirmation."""
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
        """Complete the booking."""
        result = self.tools.book_appointment(
            reason=self.context.visit_reason,
            day=self.context.preferred_day,
            time=self.context.preferred_time,
        )
        
        if result.success:
            self.context.state = AgentState.COMPLETE
            confirmation = result.data.get("confirmation_number", "N/A")
            
            # Include patient info in response
            patient_name = self.context.patient.full_name.split()[0] if self.context.patient.full_name else ""
            
            logger.info(
                "appointment_booked",
                confirmation=confirmation,
                patient_id=self.context.patient.patient_id[:8] if self.context.patient.patient_id else None,
                day=self.context.preferred_day,
                time=self.context.preferred_time,
            )
            
            return AgentResponse(
                text=f"You're all set{', ' + patient_name if patient_name else ''}! "
                     f"Your appointment is {self.context.preferred_day} at {self.context.preferred_time}. "
                     f"Confirmation number {confirmation}. We'll send a reminder to your phone. See you then!",
                ended=True,
                booking=result.data,
                patient_info={
                    "patient_id": self.context.patient.patient_id,
                    "name_masked": self.context.patient.full_name[:3] + "***" if self.context.patient.full_name else None,
                    "consent_given": self.context.patient.consent_given,
                }
            )
        else:
            return AgentResponse(text=f"Sorry, there was an issue: {result.message}. Let me try again.")

    def _do_transfer(self, reason: str) -> AgentResponse:
        """Transfer to human agent."""
        self.context.state = AgentState.TRANSFERRED
        self.tools.transfer_to_human(reason)
        
        logger.info("call_transferred", reason=reason)
        
        return AgentResponse(
            text="I'll connect you with someone who can help. Please hold.",
            ended=True,
        )

    def _get_slots_summary(self) -> str:
        """Get summary of collected slots."""
        parts = []
        if self.context.patient.full_name:
            parts.append(f"patient: {self.context.patient.full_name[:3]}***")
        if self.context.visit_reason:
            parts.append(f"reason: {self.context.visit_reason}")
        if self.context.preferred_day:
            parts.append(f"day: {self.context.preferred_day}")
        if self.context.preferred_time:
            parts.append(f"time: {self.context.preferred_time}")
        return ", ".join(parts) if parts else "none collected"
