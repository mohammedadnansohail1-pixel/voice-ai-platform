"""HIPAA-compliant audit logging."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .logger import get_logger

logger = get_logger("audit")


class AuditEventType(Enum):
    """Types of auditable events."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SPEECH_RECEIVED = "speech_received"
    TRANSCRIPT_GENERATED = "transcript_generated"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TTS_GENERATED = "tts_generated"
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    ERROR = "error"
    PHI_ACCESS = "phi_access"


# PHI patterns to redact
PHI_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),  # SSN
    (r"\b\d{9}\b", "[SSN]"),  # SSN without dashes
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    (r"\b\d{16}\b", "[CARD]"),  # Credit card
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD]"),
]


def redact_phi(text: str) -> str:
    """Redact PHI patterns from text."""
    result = text
    for pattern, replacement in PHI_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


@dataclass
class AuditEvent:
    """Audit event record."""
    event_type: AuditEventType
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    channel: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "channel": self.channel,
            "data": self.data,
        }


class AuditLogger:
    """HIPAA-compliant audit logger."""
    
    def __init__(
        self,
        enabled: bool = True,
        redact_phi: bool = True,
        audit_path: str = "logs/audit",
    ) -> None:
        self.enabled = enabled
        self.redact_phi_enabled = redact_phi
        self.audit_path = Path(audit_path)
        
        if self.enabled:
            self.audit_path.mkdir(parents=True, exist_ok=True)
    
    def _get_audit_file(self) -> Path:
        """Get today's audit file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.audit_path / f"audit_{date_str}.jsonl"
    
    def _redact_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact PHI from audit data."""
        if not self.redact_phi_enabled:
            return data
        
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = redact_phi(value)
            elif isinstance(value, dict):
                result[key] = self._redact_data(value)
            else:
                result[key] = value
        return result
    
    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        if not self.enabled:
            return
        
        # Redact PHI if enabled
        if self.redact_phi_enabled:
            event.data = self._redact_data(event.data)
        
        # Write to file
        audit_file = self._get_audit_file()
        with open(audit_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        
        # Also log via structlog for real-time visibility
        logger.info(
            "audit_event",
            event_type=event.event_type.value,
            session_id=event.session_id,
        )
    
    def session_start(self, session_id: str, channel: str, **kwargs: Any) -> None:
        """Log session start."""
        self.log(AuditEvent(
            event_type=AuditEventType.SESSION_START,
            session_id=session_id,
            channel=channel,
            data=kwargs,
        ))
    
    def session_end(self, session_id: str, duration_s: float, **kwargs: Any) -> None:
        """Log session end."""
        self.log(AuditEvent(
            event_type=AuditEventType.SESSION_END,
            session_id=session_id,
            data={"duration_s": duration_s, **kwargs},
        ))
    
    def transcript(self, session_id: str, text: str, **kwargs: Any) -> None:
        """Log transcript generated."""
        self.log(AuditEvent(
            event_type=AuditEventType.TRANSCRIPT_GENERATED,
            session_id=session_id,
            data={"text": text, **kwargs},
        ))
    
    def error(self, session_id: str, error: str, **kwargs: Any) -> None:
        """Log error."""
        self.log(AuditEvent(
            event_type=AuditEventType.ERROR,
            session_id=session_id,
            data={"error": error, **kwargs},
        ))
