"""
Inbound Agent - Handles incoming patient calls.

Hybrid architecture:
- State machine: Deterministic flow control (HIPAA-auditable)
- LLM: Natural response generation
- Guardrails: Safety checks (emergency, crisis, transfer)

Flow:
1. Greeting → Consent collection (HIPAA)
2. Patient info (name, DOB, phone)
3. Appointment booking (reason, day, time)
4. Confirmation → Complete
"""

import re
import uuid
from typing import Any, Optional

from ...core.config import Config
from ...core.types import LLMMessage
from ...logging import get_logger, AuditLogger
from ...conversation.guardrails import GuardrailAction
from ..base import BaseAgent, AgentResponse
from ..states import InboundAgentState, INBOUND_STATE_MACHINE, StateConfig
from ..context import InboundAgentContext, PatientInfo, AppointmentInfo
from ..events import EventBus, EventTypes
from ..checkpoint import CheckpointService
from ..tools import AppointmentTools
from ..extractors.slot_extractor import SlotExtractor, ExtractedSlots, ConfirmationType
from ..extractors.llm_slot_extractor import LLMSlotExtractor
from .response_generator import ResponseGenerator, GeneratedResponse

logger = get_logger("agent.inbound")


class InboundAgent(BaseAgent[InboundAgentContext, InboundAgentState]):
    """
    Voice AI agent for incoming patient calls.
    
    Hybrid approach:
    - State machine controls flow (deterministic, auditable)
    - LLM generates natural responses
    - Guardrails handle safety (emergency, crisis, transfer)
    """
    
    def __init__(
        self,
        config: Config,
        clinic_name: str = "Sunrise Medical",
        available_slots: Optional[dict[str, list[str]]] = None,
        event_bus: Optional[EventBus] = None,
        checkpoint_service: Optional[CheckpointService] = None,
        audit_logger: Optional[AuditLogger] = None,
        llm: Optional[Any] = None,
        session_id: Optional[str] = None,
        tenant_id: str = "default",
        use_llm_responses: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            checkpoint_service=checkpoint_service,
            audit_logger=audit_logger,
            llm=llm,
            session_id=session_id,
            tenant_id=tenant_id,
        )
        
        self.clinic_name = clinic_name
        self.extractor = SlotExtractor()
        self.tools = AppointmentTools(clinic_name, available_slots)
        self.use_llm_responses = use_llm_responses and llm is not None
        
        # Response generator (LLM + guardrails)
        self.response_gen = ResponseGenerator(
            llm=llm,
            clinic_name=clinic_name,
        ) if llm else None
        
        # LLM slot extractor (optional, for implicit preferences)
        self._llm_extractor: Optional[LLMSlotExtractor] = None
        
        # Tracking
        self._confirmation_attempts = 0
        self._last_offered_time: Optional[str] = None
        
        logger.info(
            "inbound_agent_initialized",
            clinic=clinic_name,
            use_llm=self.use_llm_responses,
        )
    
    @property
    def llm_extractor(self) -> LLMSlotExtractor:
        """Lazy init LLM extractor."""
        if self._llm_extractor is None:
            self._llm_extractor = LLMSlotExtractor()
        return self._llm_extractor
    
    # =========================================================================
    # BaseAgent Abstract Methods
    # =========================================================================
    
    def get_agent_type(self) -> str:
        return "inbound"
    
    def get_initial_state(self) -> InboundAgentState:
        return InboundAgentState.GREETING
    
    def get_state_config(self) -> dict[InboundAgentState, StateConfig]:
        return INBOUND_STATE_MACHINE
    
    def create_context(self) -> InboundAgentContext:
        return InboundAgentContext()
    
    async def handle_state(
        self,
        state: InboundAgentState,
        user_input: Optional[str],
    ) -> AgentResponse:
        """Route to appropriate handler based on state."""
        
        # Initial greeting (no user input yet)
        if user_input is None:
            return await self._handle_greeting()
        
        # === GUARDRAILS CHECK ===
        if self.response_gen:
            current_slots = self._get_current_slots()
            guardrail_result = self.response_gen.check_input(user_input, current_slots)
            
            if guardrail_result.action == GuardrailAction.EMERGENCY:
                logger.warning("emergency_detected", session_id=self._session_id[:8])
                await self.transition_to(InboundAgentState.TRANSFERRED)
                return AgentResponse(
                    text=guardrail_result.response,
                    state=self.get_current_state(),
                    ended=True,
                )
            
            if guardrail_result.action == GuardrailAction.TRANSFER:
                logger.info("transfer_triggered", reason=guardrail_result.reason)
                return await self._do_transfer(guardrail_result.reason)
        
        # === EXTRACT SLOTS ===
        extracted = self.extractor.extract(user_input)
        
        logger.debug(
            "input_processed",
            state=state.value,
            extracted=extracted.newly_extracted,
            session_id=self._session_id[:8],
        )
        
        # === ROUTE TO HANDLER ===
        handlers = {
            InboundAgentState.GREETING: self._handle_greeting,
            InboundAgentState.COLLECTING_CONSENT: self._handle_consent,
            InboundAgentState.COLLECTING_NAME: self._handle_name,
            InboundAgentState.COLLECTING_DOB: self._handle_dob,
            InboundAgentState.COLLECTING_PHONE: self._handle_phone,
            InboundAgentState.COLLECTING_REASON: self._handle_reason,
            InboundAgentState.COLLECTING_DAY: self._handle_day,
            InboundAgentState.CONFIRMING_DAY: self._handle_confirming_day,
            InboundAgentState.COLLECTING_TIME: self._handle_time,
            InboundAgentState.CONFIRMING: self._handle_confirming,
        }
        
        handler = handlers.get(state)
        if handler:
            if state == InboundAgentState.GREETING:
                return await handler()
            return await handler(user_input, extracted)
        
        # Terminal states
        if state in (InboundAgentState.COMPLETE, InboundAgentState.TRANSFERRED):
            return AgentResponse(
                text="Thank you for calling. Goodbye!",
                state=state.value,
                ended=True,
            )
        
        return AgentResponse(text="How can I help you?", state=state.value)
    
    # =========================================================================
    # Response Generation Helpers
    # =========================================================================
    
    def _generate_response(self, prompt_key: str, **context) -> str:
        """Generate response using LLM or fallback to template."""
        # Always add common context
        context.setdefault("clinic_name", self.clinic_name)
        context.setdefault("patient_first_name", 
                          self._context.patient.first_name if self._context else None)
        
        if self.response_gen and self.use_llm_responses:
            result = self.response_gen.generate(prompt_key, context)
            return result.text
        
        # Manual fallback (no LLM)
        return self._fallback_response(prompt_key, **context)
    
    def _fallback_response(self, prompt_key: str, **context) -> str:
        """Template-based fallback responses."""
        from .prompts import STATE_PROMPTS
        
        prompt = STATE_PROMPTS.get(prompt_key)
        if prompt:
            try:
                return prompt.fallback.format(**context)
            except KeyError:
                return prompt.fallback
        return "How can I help you?"
    
    def _get_current_slots(self) -> dict:
        """Get current slots for guardrail checking."""
        return {
            "visit_reason": self._context.appointment.reason,
            "appointment_day": self._context.appointment.day,
            "appointment_time": self._context.appointment.time,
            "patient_name": self._context.patient.full_name,
        }
    
    # =========================================================================
    # State Handlers
    # =========================================================================
    
    async def _handle_greeting(self) -> AgentResponse:
        """Initial greeting, request consent."""
        await self.transition_to(InboundAgentState.COLLECTING_CONSENT)
        
        text = self._generate_response("greeting")
        
        return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_consent(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle consent collection."""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["yes", "yeah", "sure", "okay", "ok", "consent", "agree"]):
            # Consent given
            self._context.patient.consent_given = True
            self._context.patient.consent_timestamp = self._context.updated_at
            
            await self.transition_to(InboundAgentState.COLLECTING_NAME)
            
            await self._emit_event(EventTypes.CONSENT_GIVEN, {"method": "verbal"})
            
            logger.info("consent_collected", method="verbal", session_id=self._session_id[:8])
            
            text = self._generate_response("consent_confirmed")
            return AgentResponse(text=text, state=self.get_current_state())
        
        elif any(word in input_lower for word in ["no", "don't", "refuse", "disagree"]):
            await self.transition_to(InboundAgentState.COMPLETE)
            
            text = self._generate_response("consent_refused")
            return AgentResponse(text=text, state=self.get_current_state(), ended=True)
        
        else:
            self.increment_retry()
            
            if self.is_max_retries_exceeded():
                return await self._do_transfer("consent_unclear")
            
            text = self._generate_response("collecting_consent", user_input=user_input)
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_name(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle name collection."""
        name = self._extract_name(user_input)
        
        if name:
            parts = name.split(maxsplit=1)
            self._context.patient.first_name = parts[0]
            self._context.patient.last_name = parts[1] if len(parts) > 1 else None
            
            await self.transition_to(InboundAgentState.COLLECTING_DOB)
            
            logger.info(
                "name_collected",
                name_masked=self._context.patient.full_name_masked,
                session_id=self._session_id[:8],
            )
            
            text = self._generate_response(
                "name_collected",
                patient_first_name=self._context.patient.first_name,
            )
            return AgentResponse(text=text, state=self.get_current_state())
        else:
            self.increment_retry()
            
            if self.is_max_retries_exceeded():
                return await self._do_transfer("name_unclear")
            
            text = self._generate_response("collecting_name", user_input=user_input)
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_dob(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle date of birth collection."""
        dob = self._extract_date(user_input)
        
        if dob:
            self._context.patient.date_of_birth = dob
            await self.transition_to(InboundAgentState.COLLECTING_PHONE)
            
            logger.info(
                "dob_collected",
                dob_masked=self._context.patient.dob_masked,
                session_id=self._session_id[:8],
            )
            
            text = self._generate_response("dob_collected")
            return AgentResponse(text=text, state=self.get_current_state())
        
        # Allow skip
        if any(word in user_input.lower() for word in ["skip", "prefer not", "don't want"]):
            await self.transition_to(InboundAgentState.COLLECTING_PHONE)
            text = self._generate_response("dob_collected")  # Same prompt works
            return AgentResponse(text=text, state=self.get_current_state())
        
        self.increment_retry()
        
        if self.is_max_retries_exceeded():
            await self.transition_to(InboundAgentState.COLLECTING_PHONE)
            text = self._generate_response("dob_collected")
            return AgentResponse(text=text, state=self.get_current_state())
        
        text = self._generate_response("collecting_dob", user_input=user_input)
        return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_phone(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle phone collection."""
        phone = self._extract_phone(user_input)
        
        if phone:
            self._context.patient.phone = phone
            self._context.patient.verified = True
            
            await self.transition_to(InboundAgentState.COLLECTING_REASON)
            
            logger.info(
                "phone_collected",
                phone_masked=self._context.patient.phone_masked,
                session_id=self._session_id[:8],
            )
            
            await self._emit_event(
                EventTypes.PATIENT_IDENTIFIED,
                {
                    "name_masked": self._context.patient.full_name_masked,
                    "phone_masked": self._context.patient.phone_masked,
                },
            )
            
            logger.info("patient_info_complete", session_id=self._session_id[:8])
            
            text = self._generate_response(
                "phone_collected",
                patient_first_name=self._context.patient.first_name,
            )
            return AgentResponse(text=text, state=self.get_current_state())
        else:
            self.increment_retry()
            
            if self.is_max_retries_exceeded():
                return await self._do_transfer("phone_unclear")
            
            text = self._generate_response("collecting_phone", user_input=user_input)
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_reason(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle visit reason collection."""
        if extracted.visit_reason:
            self._context.appointment.reason = extracted.visit_reason
        
        if self._context.appointment.reason:
            await self.transition_to(InboundAgentState.COLLECTING_DAY)
            
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            
            text = self._generate_response(
                "reason_collected",
                visit_reason=self._context.appointment.reason,
                available_days=days_str,
            )
            return AgentResponse(text=text, state=self.get_current_state())
        else:
            text = self._generate_response("collecting_reason")
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_day(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle day collection."""
        if extracted.preferred_day:
            self._context.appointment.day = extracted.preferred_day
        
        if self._context.appointment.day:
            availability = self.tools.check_availability(self._context.appointment.day)
            
            if availability.success:
                await self.transition_to(InboundAgentState.CONFIRMING_DAY)
                
                text = self._generate_response(
                    "confirming_day",
                    preferred_day=self._context.appointment.day,
                )
                return AgentResponse(text=text, state=self.get_current_state())
            else:
                self._context.appointment.day = None
                availability = self.tools.check_availability()
                days = list(availability.data.get("days", []))
                days_str = ", ".join(days[:3])
                
                text = self._generate_response("day_unavailable", available_days=days_str)
                return AgentResponse(text=text, state=self.get_current_state())
        else:
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            days_str = ", ".join(days[:3])
            
            text = self._generate_response("collecting_day", available_days=days_str)
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_confirming_day(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle day confirmation."""
        if extracted.confirmation == ConfirmationType.YES:
            await self.transition_to(InboundAgentState.COLLECTING_TIME)
            
            availability = self.tools.check_availability(self._context.appointment.day)
            times = availability.data.get("times", [])
            
            if times:
                self._last_offered_time = times[0]
                times_str = " or ".join(times[:2])
                
                text = self._generate_response(
                    "day_confirmed",
                    preferred_day=self._context.appointment.day,
                    available_times=times_str,
                )
                return AgentResponse(text=text, state=self.get_current_state())
            
            text = self._generate_response(
                "collecting_time",
                preferred_day=self._context.appointment.day,
                available_times="any time",
            )
            return AgentResponse(text=text, state=self.get_current_state())
        
        elif extracted.confirmation == ConfirmationType.NO:
            self._context.appointment.day = None
            await self.transition_to(InboundAgentState.COLLECTING_DAY)
            
            if extracted.preferred_day:
                self._context.appointment.day = extracted.preferred_day
                await self.transition_to(InboundAgentState.CONFIRMING_DAY)
                
                text = self._generate_response(
                    "confirming_day",
                    preferred_day=extracted.preferred_day,
                )
                return AgentResponse(text=text, state=self.get_current_state())
            
            availability = self.tools.check_availability()
            days = list(availability.data.get("days", []))
            
            text = self._generate_response("collecting_day", available_days=", ".join(days[:3]))
            return AgentResponse(text=text, state=self.get_current_state())
        
        else:
            self.increment_retry()
            
            if self.is_max_retries_exceeded():
                return await self._do_transfer("day_confirmation_unclear")
            
            text = self._generate_response(
                "confirming_day",
                preferred_day=self._context.appointment.day,
            )
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_time(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle time collection."""
        if extracted.preferred_time:
            self._context.appointment.time = extracted.preferred_time
        
        # Check if ready to confirm
        if self._is_ready_to_book():
            await self.transition_to(InboundAgentState.CONFIRMING)
            
            text = self._generate_response(
                "confirming",
                preferred_day=self._context.appointment.day,
                preferred_time=self._context.appointment.time,
                visit_reason=self._context.appointment.reason,
            )
            return AgentResponse(text=text, state=self.get_current_state())
        
        availability = self.tools.check_availability(self._context.appointment.day)
        available_times = availability.data.get("times", [])
        
        if self._context.appointment.time:
            time_lower = self._context.appointment.time.lower()
            matched_time = None
            
            for avail_time in available_times:
                if time_lower in avail_time.lower() or avail_time.lower() in time_lower:
                    matched_time = avail_time
                    break
            
            if matched_time:
                self._context.appointment.time = matched_time
                await self.transition_to(InboundAgentState.CONFIRMING)
                
                text = self._generate_response(
                    "confirming",
                    preferred_day=self._context.appointment.day,
                    preferred_time=matched_time,
                    visit_reason=self._context.appointment.reason,
                )
                return AgentResponse(text=text, state=self.get_current_state())
            else:
                self._context.appointment.time = None
                self._last_offered_time = available_times[0] if available_times else None
                times_str = " or ".join(available_times[:2])
                
                text = self._generate_response("time_unavailable", available_times=times_str)
                return AgentResponse(text=text, state=self.get_current_state())
        else:
            # Try LLM for implicit time preferences
            if available_times and self.llm:
                llm_result = self.llm_extractor.extract(
                    user_text=extracted.raw_text,
                    collecting="time",
                    available_days=[],
                    available_times=available_times,
                )
                
                if llm_result.time_preference:
                    logger.debug("llm_time_preference", preference=llm_result.time_preference)
                    
                    if llm_result.time_preference == "afternoon":
                        afternoon_times = [t for t in available_times if "PM" in t]
                        if afternoon_times:
                            self._context.appointment.time = afternoon_times[0]
                            await self.transition_to(InboundAgentState.CONFIRMING)
                            
                            text = self._generate_response(
                                "confirming",
                                preferred_day=self._context.appointment.day,
                                preferred_time=afternoon_times[0],
                                visit_reason=self._context.appointment.reason,
                            )
                            return AgentResponse(text=text, state=self.get_current_state())
                    
                    elif llm_result.time_preference == "morning":
                        morning_times = [t for t in available_times if "AM" in t]
                        if morning_times:
                            self._context.appointment.time = morning_times[0]
                            await self.transition_to(InboundAgentState.CONFIRMING)
                            
                            text = self._generate_response(
                                "confirming",
                                preferred_day=self._context.appointment.day,
                                preferred_time=morning_times[0],
                                visit_reason=self._context.appointment.reason,
                            )
                            return AgentResponse(text=text, state=self.get_current_state())
            
            self._last_offered_time = available_times[0] if available_times else None
            times_str = " or ".join(available_times[:2]) if available_times else "any time"
            
            text = self._generate_response("collecting_time", available_times=times_str)
            return AgentResponse(text=text, state=self.get_current_state())
    
    async def _handle_confirming(
        self,
        user_input: str,
        extracted: ExtractedSlots,
    ) -> AgentResponse:
        """Handle final confirmation."""
        if extracted.confirmation == ConfirmationType.YES:
            return await self._do_booking()
        
        elif extracted.confirmation == ConfirmationType.NO:
            self._context.appointment.reason = None
            self._context.appointment.day = None
            self._context.appointment.time = None
            
            await self.transition_to(InboundAgentState.COLLECTING_REASON)
            
            text = "No problem. What would you like to change?"
            return AgentResponse(text=text, state=self.get_current_state())
        
        else:
            self._confirmation_attempts += 1
            
            if self._confirmation_attempts >= 2:
                return await self._do_transfer("unclear_confirmation")
            
            text = self._generate_response(
                "confirming",
                preferred_day=self._context.appointment.day,
                preferred_time=self._context.appointment.time,
                visit_reason=self._context.appointment.reason,
            )
            return AgentResponse(text=text, state=self.get_current_state())
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    async def _do_booking(self) -> AgentResponse:
        """Complete the booking."""
        result = self.tools.book_appointment(
            reason=self._context.appointment.reason,
            day=self._context.appointment.day,
            time=self._context.appointment.time,
        )
        
        if result.success:
            await self.transition_to(InboundAgentState.COMPLETE)
            
            confirmation = result.data.get("confirmation_number", "N/A")
            self._context.appointment.confirmation_number = confirmation
            self._context.appointment.confirmed = True
            
            await self._emit_event(
                EventTypes.APPOINTMENT_BOOKED,
                {
                    "confirmation_number": confirmation,
                    "day": self._context.appointment.day,
                    "time": self._context.appointment.time,
                    "reason": self._context.appointment.reason,
                },
            )
            
            logger.info(
                "appointment_booked",
                confirmation=confirmation,
                day=self._context.appointment.day,
                time=self._context.appointment.time,
                session_id=self._session_id[:8],
            )
            
            text = self._generate_response(
                "booking_complete",
                patient_first_name=self._context.patient.first_name or "",
                preferred_day=self._context.appointment.day,
                preferred_time=self._context.appointment.time,
                confirmation_number=confirmation,
            )
            
            return AgentResponse(
                text=text,
                state=self.get_current_state(),
                ended=True,
                booking=result.data,
                patient_info={
                    "name_masked": self._context.patient.full_name_masked,
                    "consent_given": self._context.patient.consent_given,
                },
            )
        else:
            return AgentResponse(
                text=f"Sorry, there was an issue: {result.message}. Let me try again.",
                state=self.get_current_state(),
            )
    
    async def _do_transfer(self, reason: str) -> AgentResponse:
        """Transfer to human agent."""
        await self.transition_to(InboundAgentState.TRANSFERRED)
        
        self.tools.transfer_to_human(reason)
        self._context.transferred = True
        self._context.transfer_reason = reason
        
        await self._emit_event(EventTypes.CALL_TRANSFERRED, {"reason": reason})
        
        logger.info("call_transferred", reason=reason, session_id=self._session_id[:8])
        
        text = self._generate_response("transfer")
        
        return AgentResponse(text=text, state=self.get_current_state(), ended=True)
    
    # =========================================================================
    # Extraction Helpers
    # =========================================================================
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from text."""
        text = re.sub(r"^(my name is|i'm|i am|this is|it's|name is)\s*", "", text.lower()).strip()
        text = re.sub(r"[.,!?]", "", text)
        
        words = text.split()
        name_words = []
        
        for word in words[:4]:
            if len(word) < 2:
                continue
            if word in ("the", "is", "my", "its", "and", "to", "a", "an", "for"):
                continue
            if sum(c.isalpha() for c in word) >= len(word) * 0.7:
                name_words.append(word.capitalize())
        
        if name_words:
            return " ".join(name_words)
        
        fallback_words = [w.capitalize() for w in words[:3] if any(c.isalpha() for c in w)]
        if fallback_words:
            return " ".join(fallback_words)
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text."""
        text_lower = text.lower()
        
        month_map = {
            "january": "01", "jan": "01", "february": "02", "feb": "02",
            "march": "03", "mar": "03", "april": "04", "apr": "04",
            "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
            "august": "08", "aug": "08", "september": "09", "sep": "09", "sept": "09",
            "october": "10", "oct": "10", "november": "11", "nov": "11",
            "december": "12", "dec": "12",
        }
        
        patterns = [
            r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",
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
        
        year_match = re.search(r"(19\d{2}|20[0-2]\d)", text)
        if year_match:
            year = year_match.group(1)
            month = "01"
            for m_name, m_num in month_map.items():
                if m_name in text_lower:
                    month = m_num
                    break
            day_match = re.search(r"\b([1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\b", text_lower)
            day = day_match.group(1).zfill(2) if day_match else "15"
            return f"{month}/{day}/{year}"
        
        return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text."""
        digits = re.sub(r"\D", "", text)
        
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        return None
    
    def _is_ready_to_book(self) -> bool:
        """Check if all info collected for booking."""
        return all([
            self._context.appointment.reason,
            self._context.appointment.day,
            self._context.appointment.time,
            self._context.patient.is_complete(),
        ])
    
    # =========================================================================
    # Context Serialization (for checkpointing)
    # =========================================================================
    
    def _serialize_context(self) -> dict[str, Any]:
        """Serialize context for checkpointing."""
        return {
            "metadata": self._context.metadata,
            "patient": {
                "first_name": self._context.patient.first_name,
                "last_name": self._context.patient.last_name,
                "date_of_birth": self._context.patient.date_of_birth,
                "phone": self._context.patient.phone,
                "consent_given": self._context.patient.consent_given,
            },
            "appointment": {
                "reason": self._context.appointment.reason,
                "day": self._context.appointment.day,
                "time": self._context.appointment.time,
                "confirmation_number": self._context.appointment.confirmation_number,
            },
            "confirmation_attempts": self._confirmation_attempts,
            "last_offered_time": self._last_offered_time,
        }
    
    def _deserialize_context(self, data: dict[str, Any]) -> None:
        """Deserialize context from checkpoint."""
        if "metadata" in data:
            self._context.metadata = data["metadata"]
        
        if "patient" in data:
            p = data["patient"]
            self._context.patient.first_name = p.get("first_name")
            self._context.patient.last_name = p.get("last_name")
            self._context.patient.date_of_birth = p.get("date_of_birth")
            self._context.patient.phone = p.get("phone")
            self._context.patient.consent_given = p.get("consent_given", False)
        
        if "appointment" in data:
            a = data["appointment"]
            self._context.appointment.reason = a.get("reason")
            self._context.appointment.day = a.get("day")
            self._context.appointment.time = a.get("time")
            self._context.appointment.confirmation_number = a.get("confirmation_number")
        
        self._confirmation_attempts = data.get("confirmation_attempts", 0)
        self._last_offered_time = data.get("last_offered_time")
