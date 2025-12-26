"""
Multi-Agent Conversation Contexts

Dataclasses for managing agent state, patient info, and cross-agent data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid
import re


class CampaignType(Enum):
    """Outbound campaign types."""
    REMINDER_24H = "reminder_24h"
    REMINDER_2H = "reminder_2h"
    CONFIRMATION = "confirmation"
    FOLLOW_UP = "follow_up"
    NO_SHOW = "no_show"
    WELLNESS = "wellness"


class CampaignStatus(Enum):
    """Campaign execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationStatus(Enum):
    """Insurance verification status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


# =============================================================================
# Patient & Appointment Data
# =============================================================================

@dataclass
class PatientInfo:
    """Patient identification information with PHI masking."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None  # YYYY-MM-DD format
    phone: Optional[str] = None
    member_id: Optional[str] = None  # Insurance member ID
    
    # Metadata
    verified: bool = False
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    
    @property
    def full_name(self) -> Optional[str]:
        """Get full name if both parts available."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name
    
    @property
    def full_name_masked(self) -> str:
        """Get masked name for logging (HIPAA compliance)."""
        if not self.full_name:
            return "[NO NAME]"
        name = self.full_name
        if len(name) <= 2:
            return "*" * len(name)
        return name[0] + "*" * (len(name) - 2) + name[-1]
    
    @property
    def phone_masked(self) -> str:
        """Get masked phone for logging."""
        if not self.phone:
            return "[NO PHONE]"
        digits = re.sub(r'\D', '', self.phone)
        if len(digits) <= 4:
            return "*" * len(digits)
        return "*" * (len(digits) - 4) + digits[-4:]
    
    @property
    def dob_masked(self) -> str:
        """Get masked DOB for logging."""
        if not self.date_of_birth:
            return "[NO DOB]"
        # Show only year
        parts = self.date_of_birth.split("-")
        if len(parts) >= 1:
            return f"{parts[0]}-**-**"
        return "****-**-**"
    
    def is_complete(self) -> bool:
        """Check if minimum required info is collected."""
        return all([
            self.first_name,
            self.last_name,
            self.date_of_birth,
            self.phone,
        ])


@dataclass
class AppointmentInfo:
    """Appointment booking details."""
    appointment_id: Optional[str] = None
    reason: Optional[str] = None
    day: Optional[str] = None  # e.g., "Monday", "Tuesday"
    time: Optional[str] = None  # e.g., "10:00 AM", "2:30 PM"
    provider: Optional[str] = None
    location: Optional[str] = None
    duration_minutes: int = 30
    
    # Status
    confirmed: bool = False
    confirmation_number: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_datetime: Optional[datetime] = None
    
    def is_complete(self) -> bool:
        """Check if booking info is complete."""
        return all([self.reason, self.day, self.time])
    
    def generate_confirmation(self) -> str:
        """Generate a confirmation number."""
        if not self.confirmation_number:
            self.confirmation_number = f"APT-{uuid.uuid4().hex[:8].upper()}"
        return self.confirmation_number


@dataclass
class InsuranceVerificationResult:
    """Result from payer agent verification call."""
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: VerificationStatus = VerificationStatus.PENDING
    
    # Payer info
    payer_name: Optional[str] = None
    payer_phone: Optional[str] = None
    
    # Verification results
    member_verified: bool = False
    coverage_active: bool = False
    copay_amount: Optional[float] = None
    deductible: Optional[float] = None
    deductible_met: Optional[float] = None
    coinsurance_percent: Optional[float] = None
    out_of_pocket_max: Optional[float] = None
    
    # Prior authorization
    prior_auth_required: bool = False
    prior_auth_number: Optional[str] = None
    
    # Reference
    reference_number: Optional[str] = None
    rep_name: Optional[str] = None
    call_duration_s: Optional[float] = None
    
    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Raw data (for audit)
    raw_responses: list[str] = field(default_factory=list)
    
    def mark_complete(self, success: bool = True) -> None:
        """Mark verification as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = VerificationStatus.VERIFIED if success else VerificationStatus.FAILED


# =============================================================================
# Agent Contexts
# =============================================================================

@dataclass
class BaseConversationContext:
    """Base context shared by all agents."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    
    # Conversation state
    messages: list[dict] = field(default_factory=list)  # {"role": str, "content": str}
    current_state: Optional[str] = None
    previous_state: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # State machine tracking
    state_entry_time: Optional[datetime] = None
    retry_count: int = 0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.updated_at = datetime.now(timezone.utc)
    
    def get_recent_messages(self, n: int = 10) -> list[dict]:
        """Get the n most recent messages."""
        return self.messages[-n:] if self.messages else []
    
    def set_state(self, new_state: str) -> None:
        """Update state with tracking."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = datetime.now(timezone.utc)
        self.retry_count = 0
        self.updated_at = datetime.now(timezone.utc)
    
    def increment_retry(self) -> int:
        """Increment retry count and return new value."""
        self.retry_count += 1
        return self.retry_count
    
    @property
    def duration_s(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def time_in_state_s(self) -> float:
        """Get time in current state in seconds."""
        if not self.state_entry_time:
            return 0.0
        return (datetime.now(timezone.utc) - self.state_entry_time).total_seconds()


@dataclass
class InboundAgentContext(BaseConversationContext):
    """Context for inbound patient calls."""
    # Patient data
    patient: PatientInfo = field(default_factory=PatientInfo)
    
    # Appointment being booked
    appointment: AppointmentInfo = field(default_factory=AppointmentInfo)
    
    # Call metadata
    caller_id: Optional[str] = None
    call_direction: str = "inbound"
    
    # Transfer tracking
    transferred: bool = False
    transfer_reason: Optional[str] = None
    
    # Insurance verification (if triggered)
    insurance_verification_id: Optional[str] = None
    insurance_verified: bool = False


@dataclass
class OutboundAgentContext(BaseConversationContext):
    """Context for outbound campaign calls."""
    # Campaign info
    campaign_id: Optional[str] = None
    campaign_type: Optional[CampaignType] = None
    
    # Target patient
    patient: PatientInfo = field(default_factory=PatientInfo)
    
    # Existing appointment being confirmed/reminded
    appointment: AppointmentInfo = field(default_factory=AppointmentInfo)
    
    # Call metadata
    call_direction: str = "outbound"
    attempt_number: int = 1
    max_attempts: int = 3
    
    # Outcome tracking
    answered: bool = False
    voicemail_left: bool = False
    outcome: Optional[str] = None  # confirmed, rescheduled, cancelled, no_answer, voicemail
    
    # AMD (Answering Machine Detection)
    amd_result: Optional[str] = None  # human, machine, unknown


@dataclass
class PayerAgentContext(BaseConversationContext):
    """Context for payer/insurance verification calls."""
    # Patient being verified
    patient: PatientInfo = field(default_factory=PatientInfo)
    
    # Verification details
    verification: InsuranceVerificationResult = field(default_factory=InsuranceVerificationResult)
    
    # IVR navigation state
    ivr_path: list[str] = field(default_factory=list)  # Track menu selections
    dtmf_sent: list[str] = field(default_factory=list)  # DTMF tones sent
    
    # Hold tracking
    on_hold: bool = False
    hold_start_time: Optional[datetime] = None
    total_hold_time_s: float = 0.0
    
    # Speaking to rep
    speaking_to_rep: bool = False
    rep_greeting_detected: bool = False
    
    # Checkpointing (for crash recovery during long holds)
    last_checkpoint_id: Optional[str] = None
    last_checkpoint_time: Optional[datetime] = None
    checkpoint_count: int = 0
    
    # Call metadata
    call_direction: str = "outbound"
    payer_phone: Optional[str] = None
    
    def start_hold(self) -> None:
        """Mark the start of a hold period."""
        self.on_hold = True
        self.hold_start_time = datetime.now(timezone.utc)
    
    def end_hold(self) -> None:
        """Mark the end of a hold period."""
        if self.on_hold and self.hold_start_time:
            hold_duration = (datetime.now(timezone.utc) - self.hold_start_time).total_seconds()
            self.total_hold_time_s += hold_duration
        self.on_hold = False
        self.hold_start_time = None
    
    def record_checkpoint(self, checkpoint_id: str) -> None:
        """Record that a checkpoint was taken."""
        self.last_checkpoint_id = checkpoint_id
        self.last_checkpoint_time = datetime.now(timezone.utc)
        self.checkpoint_count += 1


# =============================================================================
# Campaign Management
# =============================================================================

@dataclass
class OutboundCampaign:
    """Outbound calling campaign definition."""
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_type: CampaignType = CampaignType.REMINDER_24H
    status: CampaignStatus = CampaignStatus.PENDING
    
    # Targeting
    patient_ids: list[str] = field(default_factory=list)
    appointment_ids: list[str] = field(default_factory=list)
    
    # Scheduling
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    voicemails_left: int = 0
    
    # Retry settings
    max_attempts_per_patient: int = 3
    retry_delay_minutes: int = 30
    
    # Metadata
    created_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """Mark campaign as started."""
        self.status = CampaignStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
    
    def complete(self) -> None:
        """Mark campaign as completed."""
        self.status = CampaignStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
    
    def record_outcome(self, success: bool, voicemail: bool = False) -> None:
        """Record a call outcome."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        if voicemail:
            self.voicemails_left += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate campaign success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
