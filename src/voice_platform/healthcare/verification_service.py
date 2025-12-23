"""Patient identity verification service."""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from datetime import date

from verify_core import (
    VerificationSession,
    SessionConfig,
    PatientIdentity,
    VerificationField,
    VerificationStatus,
)

from ..logging import get_logger

logger = get_logger("healthcare.verification")


@dataclass
class VerificationResult:
    """Result of a verification attempt."""
    success: bool
    field: VerificationField
    status: VerificationStatus
    message: str
    attempts_remaining: int
    is_locked: bool = False
    next_field: Optional[VerificationField] = None


class PatientVerificationService:
    """
    Patient identity verification for healthcare voice AI.
    
    Verifies caller identity using configurable factors:
    - Date of birth
    - Phone number (or last 4 digits)
    - Medical record number
    - Name (fuzzy matching)
    
    Enforces attempt limits and lockouts for security.
    """
    
    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        on_verified: Optional[Callable[[str], None]] = None,
        on_locked: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or SessionConfig(
            max_attempts_per_field=3,
            max_total_attempts=10,
            timeout_minutes=15,
            required_fields={VerificationField.DATE_OF_BIRTH},
            min_verified=2,
            lockout_minutes=30,
        )
        
        self.on_verified = on_verified
        self.on_locked = on_locked
        
        # Active sessions
        self._sessions: Dict[str, VerificationSession] = {}
        self._patients: Dict[str, PatientIdentity] = {}
        
        logger.info(
            "verification_service_initialized",
            required_fields=[f.value for f in self.config.required_fields],
            min_verified=self.config.min_verified,
        )
    
    def start_session(
        self,
        session_id: str,
        patient: PatientIdentity,
    ) -> VerificationSession:
        """Start a new verification session for a patient."""
        # Correct signature: identity, verifier (None), config
        session = VerificationSession(
            identity=patient,
            verifier=None,
            config=self.config,
        )
        
        self._sessions[session_id] = session
        self._patients[session_id] = patient
        
        logger.info(
            "verification_session_started",
            session_id=session_id[:8],
            patient_mrn=patient.mrn[:4] + "***" if patient.mrn else None,
        )
        
        return session
    
    def verify(
        self,
        session_id: str,
        field: VerificationField,
        value: str,
    ) -> VerificationResult:
        """
        Verify a single field.
        
        Returns VerificationResult with success status and next steps.
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning("verification_session_not_found", session_id=session_id[:8])
            return VerificationResult(
                success=False,
                field=field,
                status=VerificationStatus.FAILED,
                message="Session not found. Please start over.",
                attempts_remaining=0,
            )
        
        # Check if locked
        if session.is_locked:
            logger.warning("verification_session_locked", session_id=session_id[:8])
            if self.on_locked:
                self.on_locked(session_id)
            return VerificationResult(
                success=False,
                field=field,
                status=VerificationStatus.LOCKED,
                message="Too many failed attempts. Please call back later or speak with a representative.",
                attempts_remaining=0,
                is_locked=True,
            )
        
        # Check if expired
        if session.is_expired:
            logger.warning("verification_session_expired", session_id=session_id[:8])
            return VerificationResult(
                success=False,
                field=field,
                status=VerificationStatus.EXPIRED,
                message="Session expired. Please start over.",
                attempts_remaining=0,
            )
        
        # Perform verification
        result = session.verify(field, value)
        
        logger.info(
            "verification_attempt",
            session_id=session_id[:8],
            field=field.value,
            status=result.status.value,
            confidence=result.confidence,
        )
        
        # Check if fully verified
        if session.is_verified:
            logger.info("patient_verified", session_id=session_id[:8])
            if self.on_verified:
                self.on_verified(session_id)
        
        # Determine next field
        next_field = session.get_next_field()
        
        return VerificationResult(
            success=result.status == VerificationStatus.VERIFIED,
            field=field,
            status=result.status,
            message=result.message or self._get_status_message(result.status, field),
            attempts_remaining=session.remaining_attempts,
            is_locked=session.is_locked,
            next_field=next_field,
        )
    
    def is_verified(self, session_id: str) -> bool:
        """Check if session is fully verified."""
        session = self._sessions.get(session_id)
        return session.is_verified if session else False
    
    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current verification status."""
        session = self._sessions.get(session_id)
        if not session:
            return {"exists": False}
        
        return {
            "exists": True,
            "is_verified": session.is_verified,
            "is_locked": session.is_locked,
            "is_expired": session.is_expired,
            "verified_fields": [
                f.value for f, s in session.get_status().items()
                if s == VerificationStatus.VERIFIED
            ],
            "remaining_attempts": session.remaining_attempts,
            "next_field": session.get_next_field().value if session.get_next_field() else None,
        }
    
    def end_session(self, session_id: str) -> None:
        """End and cleanup a verification session."""
        self._sessions.pop(session_id, None)
        self._patients.pop(session_id, None)
        logger.info("verification_session_ended", session_id=session_id[:8])
    
    def _get_status_message(self, status: VerificationStatus, field: VerificationField) -> str:
        """Get user-friendly message for verification status."""
        messages = {
            VerificationStatus.VERIFIED: f"Thank you, {field.value.replace('_', ' ')} verified.",
            VerificationStatus.FAILED: f"That doesn't match our records. Please try again.",
            VerificationStatus.LOCKED: "Too many attempts. Please speak with a representative.",
            VerificationStatus.EXPIRED: "Session expired. Please start over.",
        }
        return messages.get(status, "Please try again.")
    
    def get_verification_prompt(self, field: VerificationField) -> str:
        """Get prompt to ask user for verification field."""
        prompts = {
            VerificationField.DATE_OF_BIRTH: "For security, can you please verify your date of birth?",
            VerificationField.PHONE: "Can you confirm the phone number on your account?",
            VerificationField.LAST_NAME: "Can you spell your last name for me?",
            VerificationField.FIRST_NAME: "And your first name?",
            VerificationField.MRN: "Do you have your medical record number?",
            VerificationField.SSN_LAST_4: "Can you provide the last four digits of your social security number?",
            VerificationField.ADDRESS_ZIP: "What's the zip code on your account?",
        }
        return prompts.get(field, f"Can you verify your {field.value.replace('_', ' ')}?")
