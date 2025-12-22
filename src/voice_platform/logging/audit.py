"""HIPAA-compliant audit logging."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .logger import get_logger

logger = get_logger("audit")


class AuditEventType(Enum):
    """Types of auditable events."""
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    SPEECH_RECEIVED = "speech_received"
    RESPONSE_GENERATED = "response_generated"
    INTENT_DETECTED = "intent_detected"
    SLOT_FILLED = "slot_filled"
    ACTION_EXECUTED = "action_executed"
    INTEGRATION_CALLED = "integration_called"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Represents an auditable event."""
    event_type: AuditEventType
    tenant_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    caller_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    
    def to_dict(self, redact_phi: bool = True) -> dict:
        """Convert to dictionary, optionally redacting PHI."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
        }
        
        if self.caller_id:
            result["caller_id"] = self._redact_phone(self.caller_id) if redact_phi else self.caller_id
        
        if self.data:
            result["data"] = self._redact_phi_fields(self.data) if redact_phi else self.data
        
        return result
    
    def _redact_phone(self, phone: str) -> str:
        """Redact phone number, keeping last 4 digits."""
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 4:
            return f"***-***-{digits[-4:]}"
        return "***"
    
    def _redact_phi_fields(self, data: dict) -> dict:
        """Redact known PHI fields."""
        phi_patterns = [
            "ssn", "social_security", "dob", "date_of_birth", "birth_date",
            "address", "street", "medical_record", "mrn", "insurance_id",
            "credit_card", "bank_account"
        ]
        
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in phi_patterns):
                result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self._redact_phi_fields(value)
            else:
                result[key] = value
        return result


class AuditLogger:
    """HIPAA-compliant audit logger."""
    
    def __init__(self, path: str = "logs/audit", redact_phi: bool = True):
        self.path = Path(path)
        self.redact_phi = redact_phi
        self.path.mkdir(parents=True, exist_ok=True)
    
    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        event_dict = event.to_dict(redact_phi=self.redact_phi)
        
        # Log to structured logger
        logger.info(
            "audit_event",
            **event_dict
        )
        
        # Also write to daily audit file
        date_str = event.timestamp.strftime("%Y-%m-%d")
        audit_file = self.path / f"audit_{date_str}.jsonl"
        
        with open(audit_file, "a") as f:
            f.write(json.dumps(event_dict) + "\n")
    
    def log_call_start(self, tenant_id: str, session_id: str, caller_id: Optional[str] = None, channel: str = "unknown") -> None:
        """Log call start event."""
        self.log(AuditEvent(
            event_type=AuditEventType.CALL_STARTED,
            tenant_id=tenant_id,
            session_id=session_id,
            caller_id=caller_id,
            data={"channel": channel}
        ))
    
    def log_call_end(self, tenant_id: str, session_id: str, duration_seconds: float, outcome: str = "completed") -> None:
        """Log call end event."""
        self.log(AuditEvent(
            event_type=AuditEventType.CALL_ENDED,
            tenant_id=tenant_id,
            session_id=session_id,
            data={"duration_seconds": duration_seconds, "outcome": outcome}
        ))
