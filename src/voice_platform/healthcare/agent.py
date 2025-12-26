"""Healthcare-enhanced conversation agent."""
from typing import Optional, Dict, Any
from datetime import date

from verify_core import PatientIdentity

from .appointment_service import AppointmentService, AppointmentResult, AppointmentStage
from .extraction_pipeline import HealthcareExtractionPipeline
from ..agent.base import AgentResponse, Stage
from ..logging import get_logger

logger = get_logger("healthcare.agent")


class HealthcareConversationAgent:
    """
    Healthcare-specific conversation agent.
    
    Wraps AppointmentService with conversation interface compatible
    with existing voice platform.
    
    Example:
        agent = HealthcareConversationAgent(
            clinic_name="Sunrise Medical",
            require_verification=True,
        )
        
        # Start conversation
        response = agent.start(patient_data=PatientIdentity(...))
        print(response.message)
        
        # Process utterances
        response = agent.process("I need to see a doctor about my headaches")
        print(response.slots)  # {"visit_reason": "headaches", ...}
    """
    
    def __init__(
        self,
        clinic_name: str = "Healthcare Clinic",
        require_verification: bool = True,
        review_confidence_threshold: float = 0.8,
        available_slots: Optional[list] = None,
    ):
        self.clinic_name = clinic_name
        
        self.service = AppointmentService(
            clinic_name=clinic_name,
            require_verification=require_verification,
            review_confidence_threshold=review_confidence_threshold,
            available_slots=available_slots,
        )
        
        self.extraction = HealthcareExtractionPipeline()
        
        self._session_id: Optional[str] = None
        self._patient_data: Optional[PatientIdentity] = None
        
        logger.info("healthcare_agent_initialized", clinic=clinic_name)
    
    def start(
        self,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_data: Optional[PatientIdentity] = None,
    ) -> AgentResponse:
        """Start a new healthcare conversation."""
        import uuid
        self._session_id = session_id or str(uuid.uuid4())
        self._patient_data = patient_data
        
        result = self.service.start(
            session_id=self._session_id,
            patient_id=patient_id,
            patient_data=patient_data,
        )
        
        return self._to_agent_response(result)
    
    def process(self, user_input: str) -> AgentResponse:
        """Process user input through healthcare pipeline."""
        if not self._session_id:
            # Auto-start if needed
            self.start()
        
        result = self.service.process(self._session_id, user_input)
        return self._to_agent_response(result)
    
    def _to_agent_response(self, result: AppointmentResult) -> AgentResponse:
        """Convert AppointmentResult to AgentResponse."""
        # Map stages
        stage_map = {
            AppointmentStage.GREETING: Stage.GREETING,
            AppointmentStage.VERIFICATION: Stage.COLLECTING,
            AppointmentStage.COLLECTING: Stage.COLLECTING,
            AppointmentStage.CONFIRMING: Stage.CONFIRMING,
            AppointmentStage.REVIEWING: Stage.CONFIRMING,
            AppointmentStage.BOOKING: Stage.CONFIRMING,
            AppointmentStage.COMPLETE: Stage.DONE,
            AppointmentStage.FAILED: Stage.DONE,
        }
        
        stage = stage_map.get(result.state.stage, Stage.COLLECTING)
        
        # Build slots dict
        slots = {
            "visit_reason": result.state.visit_reason,
            "appointment_day": result.state.preferred_day,
            "appointment_time": result.state.preferred_time,
            "department": result.state.department,
            "provider_name": result.state.provider_name,
            "patient_verified": result.state.patient_verified,
        }
        
        # Add medical info
        if result.state.medications:
            slots["medications"] = result.state.medications
        if result.state.conditions:
            slots["conditions"] = result.state.conditions
        
        # Clean None values
        slots = {k: v for k, v in slots.items() if v is not None}
        
        missing = result.state.get_missing_fields()
        
        return AgentResponse(
            message=result.message,
            slots=slots,
            stage=stage,
            ready_to_book=len(missing) == 0 and result.state.patient_verified,
            ended=result.ended,
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        if not self._session_id:
            return {}
        
        state = self.service._sessions.get(self._session_id)
        if state:
            return state.to_dict()
        return {}
    
    def end(self) -> None:
        """End the conversation."""
        if self._session_id:
            self.service.end_session(self._session_id)
            self._session_id = None
