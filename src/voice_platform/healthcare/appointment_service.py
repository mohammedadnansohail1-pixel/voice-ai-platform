"""Main appointment scheduling orchestrator."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from enum import Enum

from verify_core import PatientIdentity, VerificationField
from review_core import ReviewType
from domain_healthcare import (
    FHIRAppointment,
    FHIRMapper,
    AppointmentStatus,
)

from .config import HealthcareConfig, load_healthcare_config
from .extraction_pipeline import HealthcareExtractionPipeline, ExtractionOutput
from .verification_service import PatientVerificationService
from .review_service import HealthcareReviewService
from ..logging import get_logger

logger = get_logger("healthcare.appointment")


class AppointmentStage(str, Enum):
    """Appointment flow stages."""
    GREETING = "greeting"
    VERIFICATION = "verification"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    REVIEWING = "reviewing"  # Waiting for human review
    BOOKING = "booking"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class AppointmentState:
    """Current state of appointment booking."""
    stage: AppointmentStage = AppointmentStage.GREETING
    
    # Patient info
    patient_id: Optional[str] = None
    patient_verified: bool = False
    verified_fields: List[str] = field(default_factory=list)
    
    # Appointment details
    visit_reason: Optional[str] = None
    appointment_type: Optional[str] = None
    preferred_day: Optional[str] = None
    preferred_time: Optional[str] = None
    department: Optional[str] = None
    provider_name: Optional[str] = None
    
    # Extracted entities
    medications: List[Dict] = field(default_factory=list)
    conditions: List[Dict] = field(default_factory=list)
    
    # Review status
    pending_review_id: Optional[str] = None
    review_approved: bool = False
    
    # Booking result
    appointment_id: Optional[str] = None
    confirmation_number: Optional[str] = None
    
    # Confidence tracking
    extraction_confidence: float = 0.0
    overall_confidence: float = 0.0
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields."""
        required = ["visit_reason", "preferred_day", "preferred_time"]
        missing = []
        for field_name in required:
            if not getattr(self, field_name):
                missing.append(field_name)
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "patient_verified": self.patient_verified,
            "visit_reason": self.visit_reason,
            "appointment_type": self.appointment_type,
            "preferred_day": self.preferred_day,
            "preferred_time": self.preferred_time,
            "department": self.department,
            "provider_name": self.provider_name,
            "medications": self.medications,
            "conditions": self.conditions,
            "missing_fields": self.get_missing_fields(),
        }


@dataclass
class AppointmentResult:
    """Result of appointment service operation."""
    success: bool
    message: str
    state: AppointmentState
    next_prompt: Optional[str] = None
    requires_input: bool = True
    ended: bool = False
    fhir_appointment: Optional[Dict] = None


class AppointmentService:
    """
    Orchestrates the complete appointment scheduling flow.
    
    Integrates:
    - Entity extraction (symptoms, medications, dates/times)
    - Patient verification (DOB, phone, MRN)
    - Human-in-the-loop review
    - FHIR appointment creation
    
    All settings are config-driven via HealthcareConfig.
    """
    
    def __init__(
        self,
        config: Optional[HealthcareConfig] = None,
        config_path: Optional[str] = None,
        # Legacy parameters for backward compatibility
        clinic_name: Optional[str] = None,
        require_verification: Optional[bool] = None,
        review_confidence_threshold: Optional[float] = None,
        available_slots: Optional[List[tuple]] = None,
    ):
        # Load config
        if config:
            self.config = config
        elif config_path:
            self.config = load_healthcare_config(config_path)
        else:
            self.config = load_healthcare_config()
        
        # Override with legacy parameters if provided
        if clinic_name:
            self.config.clinic.name = clinic_name
        if require_verification is not None:
            self.config.verification.enabled = require_verification
        if review_confidence_threshold is not None:
            self.config.review.confidence_threshold = review_confidence_threshold
        if available_slots:
            self.config.available_slots = available_slots
        
        # Initialize components using config
        self.extraction = HealthcareExtractionPipeline(
            enable_medical_ner=self.config.extraction.enable_medical_ner,
            enable_clinical_extraction=self.config.extraction.enable_clinical_extraction,
            enable_phi_detection=self.config.extraction.enable_phi_detection,
        )
        
        if self.config.verification.enabled:
            self.verification = PatientVerificationService()
        else:
            self.verification = None
        
        self.review = HealthcareReviewService(
            confidence_threshold=self.config.review.confidence_threshold,
            critical_threshold=self.config.review.critical_threshold,
            sample_rate=self.config.review.sample_rate,
            auto_approve_on_expiry=self.config.review.auto_approve_on_expiry,
        )
        
        self.fhir_mapper = FHIRMapper()
        
        # Active sessions
        self._sessions: Dict[str, AppointmentState] = {}
        self._patient_data: Dict[str, PatientIdentity] = {}
        
        logger.info(
            "appointment_service_initialized",
            clinic=self.config.clinic.name,
            verification=self.config.verification.enabled,
            slots_count=len(self.config.available_slots),
        )
    
    @property
    def clinic_name(self) -> str:
        """Get clinic name from config."""
        return self.config.clinic.name
    
    @property
    def available_slots(self) -> List[tuple]:
        """Get available slots from config."""
        return self.config.available_slots
    
    def start(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        patient_data: Optional[PatientIdentity] = None,
    ) -> AppointmentResult:
        """Start a new appointment booking session."""
        state = AppointmentState(patient_id=patient_id)
        self._sessions[session_id] = state
        
        if patient_data:
            self._patient_data[session_id] = patient_data
        
        # Determine initial stage
        if self.config.verification.enabled and patient_data:
            state.stage = AppointmentStage.VERIFICATION
            self.verification.start_session(session_id, patient_data)
            
            next_field = VerificationField.DATE_OF_BIRTH
            prompt = self.verification.get_verification_prompt(next_field)
            
            greeting = (
                f"Thank you for calling {self.clinic_name}. "
                f"Before we proceed, I need to verify your identity. {prompt}"
            )
        else:
            state.stage = AppointmentStage.COLLECTING
            greeting = (
                f"Thank you for calling {self.clinic_name}. "
                "I can help you schedule an appointment. What brings you in today?"
            )
        
        logger.info(
            "appointment_session_started",
            session_id=session_id[:8],
            stage=state.stage.value,
        )
        
        return AppointmentResult(
            success=True,
            message=greeting,
            state=state,
            requires_input=True,
        )
    
    def process(self, session_id: str, user_input: str) -> AppointmentResult:
        """Process user input and advance the appointment flow."""
        state = self._sessions.get(session_id)
        if not state:
            return AppointmentResult(
                success=False,
                message="Session not found. Please start over.",
                state=AppointmentState(stage=AppointmentStage.FAILED),
                ended=True,
            )
        
        # Log with PHI redaction
        safe_input = self.extraction.get_safe_log_text(user_input)
        logger.info("processing_input", session_id=session_id[:8], input=safe_input)
        
        # Route based on stage
        if state.stage == AppointmentStage.VERIFICATION:
            return self._handle_verification(session_id, state, user_input)
        elif state.stage == AppointmentStage.COLLECTING:
            return self._handle_collecting(session_id, state, user_input)
        elif state.stage == AppointmentStage.CONFIRMING:
            return self._handle_confirming(session_id, state, user_input)
        elif state.stage == AppointmentStage.REVIEWING:
            return self._handle_reviewing(session_id, state, user_input)
        else:
            return AppointmentResult(
                success=False,
                message="I'm sorry, something went wrong. Let me transfer you.",
                state=state,
                ended=True,
            )
    
    def _handle_verification(
        self,
        session_id: str,
        state: AppointmentState,
        user_input: str,
    ) -> AppointmentResult:
        """Handle verification stage."""
        # Extract potential verification info
        extraction = self.extraction.process(user_input)
        
        # Try to verify based on what we extracted or raw input
        status = self.verification.get_status(session_id)
        next_field = VerificationField(status.get("next_field", "date_of_birth"))
        
        result = self.verification.verify(session_id, next_field, user_input)
        
        if result.is_locked:
            state.stage = AppointmentStage.FAILED
            return AppointmentResult(
                success=False,
                message="I'm sorry, we've had too many failed attempts. Please call back or speak with a representative.",
                state=state,
                ended=True,
            )
        
        if result.success:
            state.verified_fields.append(next_field.value)
        
        # Check if fully verified
        if self.verification.is_verified(session_id):
            state.patient_verified = True
            state.stage = AppointmentStage.COLLECTING
            return AppointmentResult(
                success=True,
                message="Thank you, your identity has been verified. How can I help you today? What brings you in?",
                state=state,
                requires_input=True,
            )
        
        # Need more verification
        if result.next_field:
            prompt = self.verification.get_verification_prompt(result.next_field)
            if result.success:
                message = f"Thank you. {prompt}"
            else:
                message = f"{result.message} {prompt}"
            
            return AppointmentResult(
                success=True,
                message=message,
                state=state,
                requires_input=True,
            )
        
        return AppointmentResult(
            success=True,
            message=result.message,
            state=state,
            requires_input=True,
        )
    
    def _handle_collecting(
        self,
        session_id: str,
        state: AppointmentState,
        user_input: str,
    ) -> AppointmentResult:
        """Handle information collection stage."""
        # Run extraction pipeline
        extraction = self.extraction.process(user_input)
        
        # Update state with extracted info
        slots = extraction.to_slots()
        if slots.get("visit_reason"):
            state.visit_reason = slots["visit_reason"]
        if slots.get("appointment_type"):
            state.appointment_type = slots["appointment_type"]
        if slots.get("appointment_day") or slots.get("preferred_day"):
            state.preferred_day = slots.get("appointment_day") or slots.get("preferred_day")
        if slots.get("appointment_time") or slots.get("preferred_time"):
            state.preferred_time = slots.get("appointment_time") or slots.get("preferred_time")
        if slots.get("department"):
            state.department = slots["department"]
        if slots.get("provider_name"):
            state.provider_name = slots["provider_name"]
        
        # Also look at base entities for visit reason from symptoms/conditions
        if not state.visit_reason:
            for entity in extraction.medical_entities.entities:
                if entity.entity_type.value.lower() in ["symptom", "condition"]:
                    state.visit_reason = entity.value
                    break
        
        # Store medical info
        state.medications = extraction.medications
        state.conditions = extraction.conditions
        state.extraction_confidence = self._calculate_confidence(extraction)
        
        # Check what's missing
        missing = state.get_missing_fields()
        
        if not missing:
            # Ready to confirm
            state.stage = AppointmentStage.CONFIRMING
            return AppointmentResult(
                success=True,
                message=self._build_confirmation_message(state),
                state=state,
                requires_input=True,
            )
        
        # Need more info
        prompt = self._get_next_prompt(missing, state)
        return AppointmentResult(
            success=True,
            message=prompt,
            state=state,
            requires_input=True,
        )
    
    def _handle_confirming(
        self,
        session_id: str,
        state: AppointmentState,
        user_input: str,
    ) -> AppointmentResult:
        """Handle confirmation stage."""
        input_lower = user_input.lower()
        
        # Check for confirmation
        confirms = ["yes", "yeah", "correct", "right", "confirm", "book", "sounds good", "perfect", "ok", "sure", "yep"]
        denies = ["no", "nope", "wrong", "change", "actually", "wait", "different"]
        
        if any(d in input_lower for d in denies):
            # User wants to change something
            state.stage = AppointmentStage.COLLECTING
            return AppointmentResult(
                success=True,
                message="No problem. What would you like to change?",
                state=state,
                requires_input=True,
            )
        
        if any(c in input_lower for c in confirms):
            # Check if review is needed
            needs_review, review_item = self.review.check_appointment_booking(
                patient_name=state.patient_id or "Unknown",
                appointment_day=state.preferred_day,
                appointment_time=state.preferred_time,
                visit_reason=state.visit_reason,
                ai_confidence=state.extraction_confidence,
                conversation_id=session_id,
            )
            
            if needs_review:
                state.stage = AppointmentStage.REVIEWING
                state.pending_review_id = review_item.item_id
                return AppointmentResult(
                    success=True,
                    message="Thank you. I'm confirming the appointment details. One moment please...",
                    state=state,
                    requires_input=False,  # System will callback when review complete
                )
            
            # No review needed, book directly
            return self._complete_booking(session_id, state)
        
        # Unclear response
        return AppointmentResult(
            success=True,
            message=f"Just to confirm: {state.preferred_day} at {state.preferred_time} for {state.visit_reason}. Is that correct?",
            state=state,
            requires_input=True,
        )
    
    def _handle_reviewing(
        self,
        session_id: str,
        state: AppointmentState,
        user_input: str,
    ) -> AppointmentResult:
        """Handle review waiting stage."""
        # This is typically called by system, not user
        # Just acknowledge and wait
        return AppointmentResult(
            success=True,
            message="I'm still confirming the details. Thank you for your patience.",
            state=state,
            requires_input=False,
        )
    
    def on_review_complete(self, session_id: str, approved: bool, modified_content: Optional[Dict] = None) -> AppointmentResult:
        """Called when human review is complete."""
        state = self._sessions.get(session_id)
        if not state:
            return None
        
        if approved:
            if modified_content:
                # Apply modifications
                if "appointment_day" in modified_content:
                    state.preferred_day = modified_content["appointment_day"]
                if "appointment_time" in modified_content:
                    state.preferred_time = modified_content["appointment_time"]
            
            state.review_approved = True
            return self._complete_booking(session_id, state)
        else:
            state.stage = AppointmentStage.FAILED
            return AppointmentResult(
                success=False,
                message="I'm sorry, we couldn't confirm that appointment. Let me transfer you to our scheduling team.",
                state=state,
                ended=True,
            )
    
    def _complete_booking(self, session_id: str, state: AppointmentState) -> AppointmentResult:
        """Complete the appointment booking."""
        state.stage = AppointmentStage.COMPLETE
        
        # Generate confirmation number
        import uuid
        state.confirmation_number = str(uuid.uuid4())[:8].upper()
        
        # Create FHIR appointment
        fhir_apt = FHIRAppointment(
            status=AppointmentStatus.BOOKED,
            patient_id=state.patient_id,
            description=f"{state.preferred_day} at {state.preferred_time} for {state.visit_reason}",
        )
        
        logger.info(
            "appointment_booked",
            session_id=session_id[:8],
            confirmation=state.confirmation_number,
            day=state.preferred_day,
            time=state.preferred_time,
        )
        
        return AppointmentResult(
            success=True,
            message=(
                f"Your appointment is confirmed for {state.preferred_day} at {state.preferred_time}. "
                f"Your confirmation number is {state.confirmation_number}. "
                f"You'll receive a text confirmation shortly. Thank you for calling {self.clinic_name}!"
            ),
            state=state,
            ended=True,
            fhir_appointment=fhir_apt.to_dict(),
        )
    
    def _calculate_confidence(self, extraction: ExtractionOutput) -> float:
        """Calculate overall extraction confidence."""
        if not extraction.entities.entities:
            return 0.5
        
        confidences = [e.confidence for e in extraction.entities.entities]
        return sum(confidences) / len(confidences)
    
    def _build_confirmation_message(self, state: AppointmentState) -> str:
        """Build confirmation message."""
        parts = [f"Let me confirm: {state.preferred_day} at {state.preferred_time}"]
        
        if state.visit_reason:
            parts.append(f"for {state.visit_reason}")
        
        if state.department:
            parts.append(f"in {state.department}")
        
        if state.provider_name:
            parts.append(f"with {state.provider_name}")
        
        return " ".join(parts) + ". Is that correct?"
    
    def _get_next_prompt(self, missing: List[str], state: AppointmentState) -> str:
        """Get prompt for next missing field."""
        # Prioritize: visit_reason > day > time
        if "visit_reason" in missing:
            return "What brings you in today?"
        
        if "preferred_day" in missing:
            slots_by_day = {}
            for day, time in self.available_slots:
                if day not in slots_by_day:
                    slots_by_day[day] = []
                slots_by_day[day].append(time)
            
            days = list(slots_by_day.keys())[:3]
            return f"What day works best for you? We have availability on {', '.join(days)}."
        
        if "preferred_time" in missing:
            if state.preferred_day:
                times = [t for d, t in self.available_slots if d == state.preferred_day][:3]
                return f"What time works for you on {state.preferred_day}? We have {', '.join(times)} available."
            return "What time of day works best for you - morning or afternoon?"
        
        return "Is there anything else you'd like me to know?"
    
    def end_session(self, session_id: str) -> None:
        """End and cleanup session."""
        self._sessions.pop(session_id, None)
        self._patient_data.pop(session_id, None)
        if self.verification:
            self.verification.end_session(session_id)
        logger.info("appointment_session_ended", session_id=session_id[:8])
