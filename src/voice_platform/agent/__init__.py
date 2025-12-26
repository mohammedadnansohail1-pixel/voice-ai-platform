"""
Multi-Agent Voice AI Platform

Provides specialized agents for healthcare voice interactions:
- InboundAgent: Handle incoming patient calls
- OutboundAgent: Proactive patient engagement (reminders, confirmations)
- PayerAgent: Automated insurance verification

Architecture:
- BaseAgent: Abstract base with state machine, events, checkpointing
- EventBus: Redis-backed pub/sub for agent coordination
- CheckpointService: PostgreSQL-backed crash recovery
"""

# States
from .states import (
    StateConfig,
    StateMachine,
    StateSnapshot,
    InboundAgentState,
    OutboundAgentState,
    PayerAgentState,
    INBOUND_STATE_MACHINE,
    OUTBOUND_STATE_MACHINE,
    PAYER_STATE_MACHINE,
)

# Context
from .context import (
    PatientInfo,
    AppointmentInfo,
    InsuranceVerificationResult,
    BaseConversationContext,
    InboundAgentContext,
    OutboundAgentContext,
    PayerAgentContext,
    OutboundCampaign,
    CampaignType,
    CampaignStatus,
    VerificationStatus,
)

# Events
from .events import (
    VoiceAIEvent,
    EventTypes,
    EventBus,
    EventBusBackend,
    RedisEventBusBackend,
    create_event_bus,
)

# Checkpointing
from .checkpoint import (
    StateSnapshot as CheckpointSnapshot,
    CheckpointService,
    CheckpointBackend,
    PostgreSQLCheckpointBackend,
    create_checkpoint_service,
)

# Base Agent
from .base import (
    BaseAgent,
    AgentResponse,
)

# Tools (existing, reorganized)
from .tools import (
    AppointmentTools,
    ToolResponse,
    AppointmentDatabase,
)

__all__ = [
    # States
    "StateConfig",
    "StateMachine",
    "StateSnapshot",
    "InboundAgentState",
    "OutboundAgentState",
    "PayerAgentState",
    "INBOUND_STATE_MACHINE",
    "OUTBOUND_STATE_MACHINE",
    "PAYER_STATE_MACHINE",
    # Context
    "PatientInfo",
    "AppointmentInfo",
    "InsuranceVerificationResult",
    "BaseConversationContext",
    "InboundAgentContext",
    "OutboundAgentContext",
    "PayerAgentContext",
    "OutboundCampaign",
    "CampaignType",
    "CampaignStatus",
    "VerificationStatus",
    # Events
    "VoiceAIEvent",
    "EventTypes",
    "EventBus",
    "EventBusBackend",
    "RedisEventBusBackend",
    "create_event_bus",
    # Checkpointing
    "CheckpointSnapshot",
    "CheckpointService",
    "CheckpointBackend",
    "PostgreSQLCheckpointBackend",
    "create_checkpoint_service",
    # Base Agent
    "BaseAgent",
    "AgentResponse",
    # Tools
    "AppointmentTools",
    "ToolResponse",
    "AppointmentDatabase",
]

# Legacy agent (used by run_webrtc_demo.py)
from .tool_calling_agent import ToolCallingAgent
