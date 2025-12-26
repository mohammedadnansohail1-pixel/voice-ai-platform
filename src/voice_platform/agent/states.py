"""
Production-grade state machine for multi-agent voice AI.

Features:
- Validated state transitions (prevents invalid flows)
- Configurable timeouts per state
- Retry limits with fallback states
- ASR hints for improved transcription accuracy
- Metrics hooks for observability
- Checkpointing support for long-running calls
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta

from ..logging import get_logger

logger = get_logger("agent.states")


# =============================================================================
# STATE CONFIGURATION
# =============================================================================

@dataclass
class StateConfig:
    """
    Configuration for a single state in the state machine.
    
    Attributes:
        timeout_seconds: Max time in this state before timeout handling
        max_retries: Max failed attempts before fallback/escalation
        allowed_transitions: Valid next states (enforced)
        asr_hints: Words/phrases to boost in ASR for this state
        fallback_state: State to transition to on max retries
        requires_input: Whether state waits for user input
        checkpoint: Whether to persist state (for crash recovery)
        entry_message: Optional message to speak on state entry
        timeout_message: Message to speak on timeout
        metrics_tags: Additional tags for metrics
    """
    timeout_seconds: float = 30.0
    max_retries: int = 2
    allowed_transitions: List[str] = field(default_factory=list)
    asr_hints: List[str] = field(default_factory=list)
    fallback_state: Optional[str] = None
    requires_input: bool = True
    checkpoint: bool = False
    entry_message: Optional[str] = None
    timeout_message: str = "I didn't catch that. Could you please repeat?"
    metrics_tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")


# =============================================================================
# STATE ENUMS
# =============================================================================

class InboundAgentState(Enum):
    """States for inbound patient calls."""
    # Initialization
    GREETING = "greeting"
    
    # Patient info collection (HIPAA)
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
    
    # Terminal states
    COMPLETE = "complete"
    TRANSFERRED = "transferred"
    FAILED = "failed"


class OutboundAgentState(Enum):
    """States for outbound campaign calls (reminders, confirmations)."""
    # Call initiation
    INITIATING = "initiating"
    WAITING_ANSWER = "waiting_answer"
    VOICEMAIL_DETECTED = "voicemail"
    
    # Conversation
    GREETING = "greeting"
    VERIFYING_PERSON = "verifying"
    DELIVERING_MESSAGE = "delivering"
    AWAITING_RESPONSE = "awaiting"
    HANDLING_OBJECTION = "objection"
    RESCHEDULING = "rescheduling"
    WRAPPING_UP = "wrapping_up"
    
    # Terminal states
    COMPLETE = "complete"
    FAILED = "failed"
    NO_ANSWER = "no_answer"


class PayerAgentState(Enum):
    """States for insurance payer calls (verification, prior auth)."""
    # Call initiation
    INITIATING = "initiating"
    NAVIGATING_IVR = "navigating_ivr"
    WAITING_HOLD = "waiting_hold"
    
    # Conversation with rep
    SPEAKING_TO_REP = "speaking_to_rep"
    AUTHENTICATING = "authenticating"
    PROVIDING_MEMBER_INFO = "member_info"
    REQUESTING_INFO = "requesting_info"
    EXTRACTING_DATA = "extracting_data"
    CONFIRMING_DATA = "confirming_data"
    
    # Terminal states
    COMPLETE = "complete"
    FAILED = "failed"
    ESCALATE_HUMAN = "escalate"


# =============================================================================
# STATE MACHINE CONFIGURATIONS
# =============================================================================

INBOUND_STATE_MACHINE: Dict[InboundAgentState, StateConfig] = {
    InboundAgentState.GREETING: StateConfig(
        timeout_seconds=10.0,
        max_retries=1,
        allowed_transitions=["collecting_consent", "transferred"],
        asr_hints=["hello", "hi", "appointment", "schedule", "doctor"],
        requires_input=False,
        entry_message="Thank you for calling {clinic_name}. This call may be recorded for quality purposes.",
        metrics_tags={"phase": "init"},
    ),
    
    InboundAgentState.COLLECTING_CONSENT: StateConfig(
        timeout_seconds=15.0,
        max_retries=2,
        allowed_transitions=["collecting_name", "complete", "transferred"],
        asr_hints=["yes", "yeah", "sure", "okay", "no", "consent", "agree", "disagree"],
        fallback_state="transferred",
        timeout_message="Do I have your consent to continue? Please say yes or no.",
        metrics_tags={"phase": "consent", "hipaa": "true"},
    ),
    
    InboundAgentState.COLLECTING_NAME: StateConfig(
        timeout_seconds=20.0,
        max_retries=3,
        allowed_transitions=["collecting_dob", "transferred"],
        asr_hints=[],  # Names are unpredictable, no hints
        fallback_state="transferred",
        timeout_message="Could you please tell me your full name?",
        metrics_tags={"phase": "patient_info", "field": "name"},
    ),
    
    InboundAgentState.COLLECTING_DOB: StateConfig(
        timeout_seconds=20.0,
        max_retries=3,
        allowed_transitions=["collecting_phone", "transferred"],
        asr_hints=[
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "first", "second", "third", "fourth", "fifth",
        ],
        fallback_state="collecting_phone",  # DOB optional, skip on failure
        timeout_message="What is your date of birth? For example, March 15, 1985.",
        metrics_tags={"phase": "patient_info", "field": "dob"},
    ),
    
    InboundAgentState.COLLECTING_PHONE: StateConfig(
        timeout_seconds=20.0,
        max_retries=3,
        allowed_transitions=["collecting_reason", "transferred"],
        asr_hints=["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        fallback_state="transferred",
        timeout_message="What's the best phone number to reach you?",
        metrics_tags={"phase": "patient_info", "field": "phone"},
    ),
    
    InboundAgentState.COLLECTING_REASON: StateConfig(
        timeout_seconds=30.0,
        max_retries=2,
        allowed_transitions=["collecting_day", "transferred"],
        asr_hints=["checkup", "follow-up", "pain", "sick", "fever", "cough", "appointment", "doctor"],
        fallback_state="transferred",
        timeout_message="What brings you in today?",
        metrics_tags={"phase": "booking", "field": "reason"},
    ),
    
    InboundAgentState.COLLECTING_DAY: StateConfig(
        timeout_seconds=20.0,
        max_retries=2,
        allowed_transitions=["confirming_day", "collecting_day", "transferred"],
        asr_hints=[
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "today", "tomorrow", "next week",
        ],
        fallback_state="transferred",
        timeout_message="What day works best for you?",
        metrics_tags={"phase": "booking", "field": "day"},
    ),
    
    InboundAgentState.CONFIRMING_DAY: StateConfig(
        timeout_seconds=10.0,
        max_retries=2,
        allowed_transitions=["collecting_time", "collecting_day", "transferred"],
        asr_hints=["yes", "yeah", "correct", "right", "no", "wrong", "different"],
        fallback_state="collecting_day",
        timeout_message="Was that {day}? Yes or no?",
        metrics_tags={"phase": "booking", "field": "day_confirm"},
    ),
    
    InboundAgentState.COLLECTING_TIME: StateConfig(
        timeout_seconds=20.0,
        max_retries=2,
        allowed_transitions=["confirming", "collecting_time", "transferred"],
        asr_hints=[
            "morning", "afternoon", "evening",
            "nine", "ten", "eleven", "twelve", "one", "two", "three", "four",
            "am", "pm", "o'clock",
        ],
        fallback_state="transferred",
        timeout_message="What time works for you?",
        metrics_tags={"phase": "booking", "field": "time"},
    ),
    
    InboundAgentState.CONFIRMING: StateConfig(
        timeout_seconds=15.0,
        max_retries=3,
        allowed_transitions=["complete", "collecting_reason", "transferred"],
        asr_hints=["yes", "yeah", "book", "confirm", "no", "change", "cancel"],
        fallback_state="transferred",
        checkpoint=True,  # Persist before final action
        timeout_message="Should I book this appointment? Yes or no?",
        metrics_tags={"phase": "booking", "field": "confirm"},
    ),
    
    InboundAgentState.COMPLETE: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "success"},
    ),
    
    InboundAgentState.TRANSFERRED: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "transfer"},
    ),
    
    InboundAgentState.FAILED: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "failed"},
    ),
}


OUTBOUND_STATE_MACHINE: Dict[OutboundAgentState, StateConfig] = {
    OutboundAgentState.INITIATING: StateConfig(
        timeout_seconds=5.0,
        max_retries=0,
        allowed_transitions=["waiting_answer", "failed"],
        requires_input=False,
        metrics_tags={"phase": "init"},
    ),
    
    OutboundAgentState.WAITING_ANSWER: StateConfig(
        timeout_seconds=30.0,  # Ring timeout
        max_retries=0,
        allowed_transitions=["greeting", "voicemail", "no_answer"],
        requires_input=False,
        metrics_tags={"phase": "connect"},
    ),
    
    OutboundAgentState.VOICEMAIL_DETECTED: StateConfig(
        timeout_seconds=60.0,  # Time to leave message
        max_retries=0,
        allowed_transitions=["complete"],
        requires_input=False,
        entry_message="This is a reminder from {clinic_name} about your appointment on {day} at {time}. Please call us back to confirm.",
        metrics_tags={"phase": "voicemail"},
    ),
    
    OutboundAgentState.GREETING: StateConfig(
        timeout_seconds=10.0,
        max_retries=1,
        allowed_transitions=["verifying", "complete", "failed"],
        asr_hints=["hello", "yes", "speaking", "who", "what"],
        entry_message="Hello, this is an automated call from {clinic_name}.",
        metrics_tags={"phase": "connect"},
    ),
    
    OutboundAgentState.VERIFYING_PERSON: StateConfig(
        timeout_seconds=15.0,
        max_retries=2,
        allowed_transitions=["delivering", "failed"],
        asr_hints=["yes", "speaking", "that's me", "no", "wrong number"],
        fallback_state="failed",
        timeout_message="Am I speaking with {patient_name}?",
        metrics_tags={"phase": "verify"},
    ),
    
    OutboundAgentState.DELIVERING_MESSAGE: StateConfig(
        timeout_seconds=30.0,
        max_retries=1,
        allowed_transitions=["awaiting", "objection", "complete"],
        asr_hints=["okay", "got it", "wait", "question", "cancel", "reschedule"],
        requires_input=False,
        metrics_tags={"phase": "deliver"},
    ),
    
    OutboundAgentState.AWAITING_RESPONSE: StateConfig(
        timeout_seconds=15.0,
        max_retries=2,
        allowed_transitions=["complete", "objection", "rescheduling", "failed"],
        asr_hints=["yes", "confirm", "no", "cancel", "reschedule", "change", "question"],
        fallback_state="failed",
        timeout_message="Can you confirm your appointment?",
        metrics_tags={"phase": "response"},
    ),
    
    OutboundAgentState.HANDLING_OBJECTION: StateConfig(
        timeout_seconds=30.0,
        max_retries=2,
        allowed_transitions=["rescheduling", "complete", "failed"],
        asr_hints=["cancel", "reschedule", "different", "time", "day", "can't", "unable"],
        fallback_state="failed",
        metrics_tags={"phase": "objection"},
    ),
    
    OutboundAgentState.RESCHEDULING: StateConfig(
        timeout_seconds=60.0,
        max_retries=2,
        allowed_transitions=["complete", "failed"],
        asr_hints=[
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "morning", "afternoon", "nine", "ten", "eleven", "two", "three",
        ],
        fallback_state="failed",
        metrics_tags={"phase": "reschedule"},
    ),
    
    OutboundAgentState.WRAPPING_UP: StateConfig(
        timeout_seconds=10.0,
        max_retries=0,
        allowed_transitions=["complete"],
        requires_input=False,
        entry_message="Thank you. We look forward to seeing you. Goodbye!",
        metrics_tags={"phase": "wrapup"},
    ),
    
    OutboundAgentState.COMPLETE: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "success"},
    ),
    
    OutboundAgentState.FAILED: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "failed"},
    ),
    
    OutboundAgentState.NO_ANSWER: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "no_answer"},
    ),
}


PAYER_STATE_MACHINE: Dict[PayerAgentState, StateConfig] = {
    PayerAgentState.INITIATING: StateConfig(
        timeout_seconds=10.0,
        max_retries=3,  # Retry dialing
        allowed_transitions=["navigating_ivr", "failed"],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "init"},
    ),
    
    PayerAgentState.NAVIGATING_IVR: StateConfig(
        timeout_seconds=120.0,  # IVR can be long
        max_retries=5,  # Multiple menu levels
        allowed_transitions=["navigating_ivr", "waiting_hold", "speaking_to_rep", "failed"],
        asr_hints=[
            "press", "one", "two", "three", "four", "five",
            "provider", "eligibility", "claims", "representative",
        ],
        checkpoint=True,
        metrics_tags={"phase": "ivr"},
    ),
    
    PayerAgentState.WAITING_HOLD: StateConfig(
        timeout_seconds=3600.0,  # Up to 1 hour hold
        max_retries=0,
        allowed_transitions=["speaking_to_rep", "failed"],
        requires_input=False,
        checkpoint=True,  # CRITICAL: Resume after crash
        metrics_tags={"phase": "hold"},
    ),
    
    PayerAgentState.SPEAKING_TO_REP: StateConfig(
        timeout_seconds=30.0,
        max_retries=1,
        allowed_transitions=["authenticating", "escalate", "failed"],
        asr_hints=["hello", "hi", "can I help", "how may I"],
        checkpoint=True,
        metrics_tags={"phase": "conversation"},
    ),
    
    PayerAgentState.AUTHENTICATING: StateConfig(
        timeout_seconds=60.0,
        max_retries=2,
        allowed_transitions=["providing_member_info", "escalate", "failed"],
        asr_hints=["npi", "tax id", "provider", "name", "verify"],
        checkpoint=True,
        metrics_tags={"phase": "auth"},
    ),
    
    PayerAgentState.PROVIDING_MEMBER_INFO: StateConfig(
        timeout_seconds=60.0,
        max_retries=2,
        allowed_transitions=["requesting_info", "escalate", "failed"],
        asr_hints=["member", "id", "date of birth", "subscriber", "patient"],
        checkpoint=True,
        metrics_tags={"phase": "member_info"},
    ),
    
    PayerAgentState.REQUESTING_INFO: StateConfig(
        timeout_seconds=120.0,  # Rep may need time to look up
        max_retries=3,
        allowed_transitions=["extracting_data", "requesting_info", "escalate", "failed"],
        asr_hints=[
            "deductible", "copay", "coinsurance", "out of pocket",
            "prior auth", "authorization", "covered", "benefits",
        ],
        checkpoint=True,
        metrics_tags={"phase": "request"},
    ),
    
    PayerAgentState.EXTRACTING_DATA: StateConfig(
        timeout_seconds=60.0,
        max_retries=2,
        allowed_transitions=["confirming_data", "requesting_info", "escalate"],
        requires_input=True,  # Processing rep's response
        checkpoint=True,
        metrics_tags={"phase": "extract"},
    ),
    
    PayerAgentState.CONFIRMING_DATA: StateConfig(
        timeout_seconds=60.0,
        max_retries=2,
        allowed_transitions=["complete", "requesting_info", "escalate"],
        asr_hints=["correct", "yes", "no", "reference", "number", "confirmation"],
        checkpoint=True,
        metrics_tags={"phase": "confirm"},
    ),
    
    PayerAgentState.COMPLETE: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "success"},
    ),
    
    PayerAgentState.FAILED: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "failed"},
    ),
    
    PayerAgentState.ESCALATE_HUMAN: StateConfig(
        timeout_seconds=0,
        max_retries=0,
        allowed_transitions=[],
        requires_input=False,
        checkpoint=True,
        metrics_tags={"phase": "terminal", "outcome": "escalate"},
    ),
}


# =============================================================================
# STATE MACHINE MANAGER
# =============================================================================

@dataclass
class StateSnapshot:
    """Snapshot for checkpointing and crash recovery."""
    state: str
    agent_type: str
    session_id: str
    timestamp: datetime
    retry_count: int
    context_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "context_data": self.context_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSnapshot":
        return cls(
            state=data["state"],
            agent_type=data["agent_type"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            retry_count=data["retry_count"],
            context_data=data["context_data"],
        )


class StateMachine:
    """
    Production state machine with validation, timeouts, and checkpointing.
    
    Features:
    - Validates all transitions
    - Tracks retry counts per state
    - Handles timeouts
    - Supports checkpointing for crash recovery
    - Emits metrics
    """
    
    def __init__(
        self,
        agent_type: str,
        state_config: Dict[Enum, StateConfig],
        initial_state: Enum,
        session_id: str,
        on_checkpoint: Optional[Callable[[StateSnapshot], None]] = None,
        on_metrics: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.agent_type = agent_type
        self.state_config = state_config
        self.current_state = initial_state
        self.session_id = session_id
        self.on_checkpoint = on_checkpoint
        self.on_metrics = on_metrics
        
        # Tracking
        self.retry_counts: Dict[str, int] = {}
        self.state_entry_time: datetime = datetime.utcnow()
        self.transition_history: List[Dict[str, Any]] = []
        
        # Validate config on init
        self._validate_config()
        
        logger.info(
            "state_machine_initialized",
            agent_type=agent_type,
            initial_state=initial_state.value,
            session_id=session_id[:8],
        )
    
    def _validate_config(self) -> None:
        """Validate state machine configuration."""
        all_states = {s.value for s in self.state_config.keys()}
        
        for state, config in self.state_config.items():
            # Check all transitions point to valid states
            for transition in config.allowed_transitions:
                if transition not in all_states:
                    raise ValueError(
                        f"State {state.value} has invalid transition to '{transition}'"
                    )
            
            # Check fallback state is valid
            if config.fallback_state and config.fallback_state not in all_states:
                raise ValueError(
                    f"State {state.value} has invalid fallback_state '{config.fallback_state}'"
                )
        
        # Check for unreachable states (except initial)
        reachable = set()
        to_visit = [list(self.state_config.keys())[0].value]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            reachable.add(current)
            
            for state, config in self.state_config.items():
                if state.value == current:
                    to_visit.extend(config.allowed_transitions)
                    break
        
        unreachable = all_states - reachable
        if unreachable:
            logger.warning(
                "unreachable_states_detected",
                states=list(unreachable),
            )
    
    @property
    def config(self) -> StateConfig:
        """Get current state's configuration."""
        return self.state_config[self.current_state]
    
    def can_transition_to(self, target_state: Enum) -> bool:
        """Check if transition to target state is allowed."""
        return target_state.value in self.config.allowed_transitions
    
    def transition_to(
        self,
        target_state: Enum,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition to a new state.
        
        Returns True if transition succeeded, False if invalid.
        """
        if not self.can_transition_to(target_state):
            logger.warning(
                "invalid_transition_attempted",
                from_state=self.current_state.value,
                to_state=target_state.value,
                allowed=self.config.allowed_transitions,
            )
            return False
        
        # Record transition
        transition_record = {
            "from": self.current_state.value,
            "to": target_state.value,
            "timestamp": datetime.utcnow().isoformat(),
            "time_in_state_ms": (datetime.utcnow() - self.state_entry_time).total_seconds() * 1000,
        }
        self.transition_history.append(transition_record)
        
        # Emit metrics
        if self.on_metrics:
            self.on_metrics("state_transition", {
                **transition_record,
                "session_id": self.session_id,
                "agent_type": self.agent_type,
                **self.config.metrics_tags,
            })
        
        # Update state
        old_state = self.current_state
        self.current_state = target_state
        self.state_entry_time = datetime.utcnow()
        
        # Reset retry count for new state
        self.retry_counts[target_state.value] = 0
        
        logger.info(
            "state_transition",
            from_state=old_state.value,
            to_state=target_state.value,
            session_id=self.session_id[:8],
        )
        
        # Checkpoint if configured
        if self.state_config[target_state].checkpoint and self.on_checkpoint:
            snapshot = StateSnapshot(
                state=target_state.value,
                agent_type=self.agent_type,
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                retry_count=0,
                context_data=context_data or {},
            )
            self.on_checkpoint(snapshot)
        
        return True
    
    def increment_retry(self) -> bool:
        """
        Increment retry count for current state.
        
        Returns True if under limit, False if limit exceeded.
        """
        state_value = self.current_state.value
        self.retry_counts[state_value] = self.retry_counts.get(state_value, 0) + 1
        
        current_retries = self.retry_counts[state_value]
        max_retries = self.config.max_retries
        
        if current_retries > max_retries:
            logger.warning(
                "max_retries_exceeded",
                state=state_value,
                retries=current_retries,
                max=max_retries,
                session_id=self.session_id[:8],
            )
            return False
        
        logger.debug(
            "retry_incremented",
            state=state_value,
            retries=current_retries,
            max=max_retries,
        )
        return True
    
    def get_retry_count(self) -> int:
        """Get current retry count for current state."""
        return self.retry_counts.get(self.current_state.value, 0)
    
    def is_timeout(self) -> bool:
        """Check if current state has timed out."""
        if self.config.timeout_seconds <= 0:
            return False
        
        elapsed = (datetime.utcnow() - self.state_entry_time).total_seconds()
        return elapsed > self.config.timeout_seconds
    
    def get_fallback_state(self, state_enum_class: type) -> Optional[Enum]:
        """Get fallback state for current state, if configured."""
        fallback = self.config.fallback_state
        if fallback:
            try:
                return state_enum_class(fallback)
            except ValueError:
                logger.error("invalid_fallback_state", fallback=fallback)
        return None
    
    def is_terminal(self) -> bool:
        """Check if current state is terminal (no transitions out)."""
        return len(self.config.allowed_transitions) == 0
    
    def get_asr_hints(self) -> List[str]:
        """Get ASR hints for current state."""
        return self.config.asr_hints
    
    def get_timeout_message(self) -> str:
        """Get timeout message for current state."""
        return self.config.timeout_message
    
    def get_entry_message(self) -> Optional[str]:
        """Get entry message for current state."""
        return self.config.entry_message
    
    def create_snapshot(self, context_data: Dict[str, Any]) -> StateSnapshot:
        """Create a snapshot for checkpointing."""
        return StateSnapshot(
            state=self.current_state.value,
            agent_type=self.agent_type,
            session_id=self.session_id,
            timestamp=datetime.utcnow(),
            retry_count=self.get_retry_count(),
            context_data=context_data,
        )
    
    @classmethod
    def restore_from_snapshot(
        cls,
        snapshot: StateSnapshot,
        state_enum_class: type,
        state_config: Dict[Enum, StateConfig],
        on_checkpoint: Optional[Callable[[StateSnapshot], None]] = None,
        on_metrics: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> "StateMachine":
        """Restore state machine from a checkpoint snapshot."""
        initial_state = state_enum_class(snapshot.state)
        
        machine = cls(
            agent_type=snapshot.agent_type,
            state_config=state_config,
            initial_state=initial_state,
            session_id=snapshot.session_id,
            on_checkpoint=on_checkpoint,
            on_metrics=on_metrics,
        )
        
        machine.retry_counts[snapshot.state] = snapshot.retry_count
        
        logger.info(
            "state_machine_restored",
            state=snapshot.state,
            session_id=snapshot.session_id[:8],
            checkpoint_time=snapshot.timestamp.isoformat(),
        )
        
        return machine