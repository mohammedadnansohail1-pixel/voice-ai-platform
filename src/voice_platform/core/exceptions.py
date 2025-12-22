"""
Voice AI Platform - Exception Hierarchy

All exceptions include error codes and details for consistent error handling.
"""

from typing import Any, Optional


class VoicePlatformError(Exception):
    """Base exception for all platform errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}


# --- Configuration Errors ---

class ConfigError(VoicePlatformError):
    """Configuration-related errors."""
    pass


class ConfigNotFoundError(ConfigError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Config not found: {path}", "CONFIG_NOT_FOUND", {"path": path})


class ConfigValidationError(ConfigError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(
            f"Config validation failed: {len(errors)} error(s)",
            "CONFIG_VALIDATION_ERROR",
            {"validation_errors": errors},
        )


# --- Model Errors ---

class ModelError(VoicePlatformError):
    """ML model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    def __init__(self, category: str, backend: Optional[str] = None) -> None:
        msg = f"Model not found: {category}" + (f"/{backend}" if backend else "")
        super().__init__(msg, "MODEL_NOT_FOUND", {"category": category, "backend": backend})


class ModelLoadError(ModelError):
    def __init__(self, category: str, backend: str, reason: str) -> None:
        super().__init__(
            f"Failed to load {category}/{backend}: {reason}",
            "MODEL_LOAD_ERROR",
            {"category": category, "backend": backend, "reason": reason},
        )


class ModelInferenceError(ModelError):
    def __init__(self, model: str, operation: str, reason: str) -> None:
        super().__init__(
            f"Inference failed for {model}.{operation}: {reason}",
            "MODEL_INFERENCE_ERROR",
            {"model": model, "operation": operation, "reason": reason},
        )


# --- Audio Errors ---

class AudioError(VoicePlatformError):
    """Audio processing errors."""
    pass


class AudioFormatError(AudioError):
    def __init__(self, expected: str, received: str) -> None:
        super().__init__(
            f"Invalid audio format. Expected {expected}, got {received}",
            "AUDIO_FORMAT_ERROR",
            {"expected": expected, "received": received},
        )


class AudioProcessingError(AudioError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(
            f"Audio processing failed at '{stage}': {reason}",
            "AUDIO_PROCESSING_ERROR",
            {"stage": stage, "reason": reason},
        )


# --- Session Errors ---

class SessionError(VoicePlatformError):
    """Session management errors."""
    pass


class SessionNotFoundError(SessionError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}", "SESSION_NOT_FOUND", {"session_id": session_id})


class SessionLimitError(SessionError):
    def __init__(self, current: int, limit: int) -> None:
        super().__init__(
            f"Session limit exceeded: {current}/{limit}",
            "SESSION_LIMIT_EXCEEDED",
            {"current": current, "limit": limit},
        )


class SessionTimeoutError(SessionError):
    def __init__(self, session_id: str, timeout_seconds: int) -> None:
        super().__init__(
            f"Session {session_id} timed out after {timeout_seconds}s",
            "SESSION_TIMEOUT",
            {"session_id": session_id, "timeout_seconds": timeout_seconds},
        )


# --- Channel Errors ---

class ChannelError(VoicePlatformError):
    """Communication channel errors."""
    pass


class ChannelNotFoundError(ChannelError):
    def __init__(self, channel_type: str, available: list[str]) -> None:
        super().__init__(
            f"Channel '{channel_type}' not found. Available: {available}",
            "CHANNEL_NOT_FOUND",
            {"channel_type": channel_type, "available": available},
        )


class ChannelConnectionError(ChannelError):
    def __init__(self, channel_type: str, reason: str) -> None:
        super().__init__(
            f"Channel '{channel_type}' connection failed: {reason}",
            "CHANNEL_CONNECTION_ERROR",
            {"channel_type": channel_type, "reason": reason},
        )


class ChannelDisconnectedError(ChannelError):
    def __init__(self, channel_type: str, session_id: str) -> None:
        super().__init__(
            f"Channel '{channel_type}' disconnected",
            "CHANNEL_DISCONNECTED",
            {"channel_type": channel_type, "session_id": session_id},
        )


# --- Conversation Errors ---

class ConversationError(VoicePlatformError):
    """Conversation logic errors."""
    pass


class FlowNotFoundError(ConversationError):
    def __init__(self, flow_name: str, search_path: str) -> None:
        super().__init__(
            f"Flow '{flow_name}' not found in {search_path}",
            "FLOW_NOT_FOUND",
            {"flow_name": flow_name, "search_path": search_path},
        )


class FlowValidationError(ConversationError):
    def __init__(self, flow_name: str, errors: list[str]) -> None:
        super().__init__(
            f"Flow '{flow_name}' validation failed",
            "FLOW_VALIDATION_ERROR",
            {"flow_name": flow_name, "errors": errors},
        )


class GuardrailTriggeredError(ConversationError):
    def __init__(self, guardrail: str, action: str, reason: str) -> None:
        super().__init__(
            f"Guardrail '{guardrail}' triggered: {reason}",
            "GUARDRAIL_TRIGGERED",
            {"guardrail": guardrail, "action": action, "reason": reason},
        )


# --- Telephony Errors ---

class TelephonyError(VoicePlatformError):
    """Phone call errors."""
    pass


class CallFailedError(TelephonyError):
    def __init__(self, call_sid: Optional[str], reason: str) -> None:
        super().__init__(f"Call failed: {reason}", "CALL_FAILED", {"call_sid": call_sid, "reason": reason})


class InvalidPhoneNumberError(TelephonyError):
    def __init__(self, phone_number: str, reason: str) -> None:
        masked = phone_number[:3] + "***" + phone_number[-2:] if len(phone_number) > 5 else "***"
        super().__init__(f"Invalid phone number ({masked}): {reason}", "INVALID_PHONE_NUMBER", {"reason": reason})


# --- HTTP Status Mapping ---

HTTP_STATUS_MAP: dict[str, int] = {
    "CONFIG_NOT_FOUND": 404,
    "CONFIG_VALIDATION_ERROR": 400,
    "MODEL_NOT_FOUND": 404,
    "MODEL_LOAD_ERROR": 503,
    "MODEL_INFERENCE_ERROR": 503,
    "AUDIO_FORMAT_ERROR": 400,
    "AUDIO_PROCESSING_ERROR": 500,
    "SESSION_NOT_FOUND": 404,
    "SESSION_LIMIT_EXCEEDED": 503,
    "SESSION_TIMEOUT": 408,
    "CHANNEL_NOT_FOUND": 404,
    "CHANNEL_CONNECTION_ERROR": 502,
    "CHANNEL_DISCONNECTED": 502,
    "FLOW_NOT_FOUND": 404,
    "FLOW_VALIDATION_ERROR": 400,
    "GUARDRAIL_TRIGGERED": 200,
    "CALL_FAILED": 502,
    "INVALID_PHONE_NUMBER": 400,
    "UNKNOWN_ERROR": 500,
}


def get_http_status(error: VoicePlatformError) -> int:
    """Get HTTP status code for an exception."""
    return HTTP_STATUS_MAP.get(error.code, 500)
