"""
Base Agent Abstract Class

All agents (Inbound, Outbound, Payer) inherit from this.
Provides state machine, event emission, checkpointing, and common utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, TypeVar, Generic
import uuid

from ..core.config import Config
from ..core.types import LLMMessage
from ..logging import get_logger, AuditLogger
from .states import StateConfig, StateMachine
from .context import BaseConversationContext
from .events import EventBus, VoiceAIEvent, EventTypes
from .checkpoint import CheckpointService, StateSnapshot

logger = get_logger("agent.base")


# =============================================================================
# Agent Response
# =============================================================================

@dataclass
class AgentResponse:
    """Response from agent processing."""
    text: str
    state: str
    ended: bool = False
    
    slots: dict[str, Any] = field(default_factory=dict)
    booking: Optional[dict] = None
    patient_info: Optional[dict] = None
    verification_result: Optional[dict] = None
    
    events_emitted: list[str] = field(default_factory=list)
    checkpoint_created: bool = False
    processing_time_ms: float = 0.0


# =============================================================================
# Type Variables
# =============================================================================

TContext = TypeVar('TContext', bound=BaseConversationContext)
TState = TypeVar('TState', bound=Enum)


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC, Generic[TContext, TState]):
    """
    Abstract base class for all voice agents.
    
    Provides:
    - State machine management with validation
    - Conversation context lifecycle
    - Event emission for cross-agent communication
    - Checkpointing for crash recovery
    - Audit logging with PHI protection
    """
    
    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        checkpoint_service: Optional[CheckpointService] = None,
        audit_logger: Optional[AuditLogger] = None,
        llm: Optional[Any] = None,
        session_id: Optional[str] = None,
        tenant_id: str = "default",
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.checkpoint_service = checkpoint_service
        self.audit = audit_logger
        self.llm = llm
        self.tenant_id = tenant_id
        
        self._context: Optional[TContext] = None
        self._state_machine: Optional[StateMachine] = None
        self._session_id = session_id or str(uuid.uuid4())
        
        self._started = False
        self._ended = False
        self._checkpoint_count = 0
    
    # =========================================================================
    # Abstract Methods
    # =========================================================================
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        pass
    
    @abstractmethod
    def get_initial_state(self) -> TState:
        """Return the initial state for this agent."""
        pass
    
    @abstractmethod
    def get_state_config(self) -> dict[TState, StateConfig]:
        """Return state machine configuration."""
        pass
    
    @abstractmethod
    def create_context(self) -> TContext:
        """Create a new context for this agent."""
        pass
    
    @abstractmethod
    async def handle_state(self, state: TState, user_input: Optional[str]) -> AgentResponse:
        """Handle processing for a specific state."""
        pass
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    async def start(self, session_id: Optional[str] = None) -> AgentResponse:
        """Initialize the agent and return initial greeting."""
        if self._started:
            raise RuntimeError("Agent already started")
        
        if session_id:
            self._session_id = session_id
        
        self._context = self.create_context()
        self._context.session_id = self._session_id
        self._context.tenant_id = self.tenant_id
        
        initial_state = self.get_initial_state()
        state_config = self.get_state_config()
        self._state_machine = StateMachine(
            agent_type=self.get_agent_type(),
            state_config=state_config,
            initial_state=initial_state,
            session_id=self._session_id,
        )  # Fixed
        
        self._context.set_state(initial_state.value)
        self._started = True
        
        logger.info(
            "agent_started",
            agent_type=self.get_agent_type(),
            session_id=self._session_id[:8],
            initial_state=initial_state.value,
        )
        
        await self._emit_event(
            EventTypes.AGENT_STARTED,
            {"initial_state": initial_state.value},
        )
        
        if self.audit:
            self.audit.session_start(self._session_id, channel=self.get_agent_type())
        
        if self.config.checkpoint.checkpoint_on_state_change:
            await self._create_checkpoint("start")
        
        return await self.handle_state(initial_state, None)
    
    async def process(self, user_input: str) -> AgentResponse:
        """Process user input and return agent response."""
        if not self._started:
            raise RuntimeError("Agent not started")
        
        if self._ended:
            raise RuntimeError("Agent has ended")
        
        start_time = datetime.now(timezone.utc)
        
        self._context.add_message("user", user_input)
        
        current_state = self._get_current_state_enum()
        
        state_config = self.get_state_config().get(current_state)
        if state_config and self._state_machine:
            if self._state_machine.is_timeout():
                logger.warning(
                    "state_timeout",
                    state=current_state.value,
                    session_id=self._session_id[:8],
                )
                if state_config.fallback_state:
                    fallback = self._parse_state(state_config.fallback_state)
                    if fallback:
                        await self.transition_to(fallback)
                        current_state = fallback
        
        response = await self.handle_state(current_state, user_input)
        
        self._context.add_message("assistant", response.text)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        response.processing_time_ms = processing_time
        
        if response.ended:
            await self.end()
        
        return response
    
    async def end(self, reason: str = "completed") -> None:
        """End the agent session."""
        if self._ended:
            return
        
        self._ended = True
        
        await self._emit_event(
            EventTypes.AGENT_ENDED,
            {
                "reason": reason,
                "final_state": self._context.current_state if self._context else None,
                "duration_s": self._context.duration_s if self._context else 0,
            },
        )
        
        await self._create_checkpoint("end")
        
        if self.audit and self._context:
            self.audit.session_end(self._session_id, duration_s=self._context.duration_s)
        
        logger.info(
            "agent_ended",
            agent_type=self.get_agent_type(),
            session_id=self._session_id[:8],
            reason=reason,
            duration_s=self._context.duration_s if self._context else 0,
        )
    
    # =========================================================================
    # State Machine Methods
    # =========================================================================
    
    async def transition_to(self, new_state: TState) -> bool:
        """Transition to a new state with validation."""
        if not self._state_machine:
            raise RuntimeError("State machine not initialized")
        
        current_state = self._get_current_state_enum()
        
        success = self._state_machine.transition_to(new_state)
        
        if success:
            self._context.set_state(new_state.value)
            
            logger.debug(
                "state_transition",
                from_state=current_state.value,
                to_state=new_state.value,
                session_id=self._session_id[:8],
            )
            
            if self.config.checkpoint.checkpoint_on_state_change:
                await self._create_checkpoint("state_change")
        else:
            logger.warning(
                "invalid_state_transition",
                from_state=current_state.value,
                to_state=new_state.value,
                session_id=self._session_id[:8],
            )
        
        return success
    
    def get_current_state(self) -> str:
        """Get current state as string."""
        return self._context.current_state if self._context else ""
    
    def _get_current_state_enum(self) -> TState:
        """Get current state as enum."""
        if not self._state_machine:
            raise RuntimeError("State machine not initialized")
        return self._state_machine.current_state
    
    def _parse_state(self, state_name: str) -> Optional[TState]:
        """Parse state name string to enum."""
        state_config = self.get_state_config()
        for state in state_config.keys():
            if state.value == state_name or state.name == state_name:
                return state
        return None
    
    def increment_retry(self) -> int:
        """Increment retry count for current state."""
        if self._state_machine:
            return self._state_machine.increment_retry()
        return 0
    
    def is_max_retries_exceeded(self) -> bool:
        """Check if max retries exceeded for current state."""
        if not self._state_machine:
            return False
        
        current_state = self._get_current_state_enum()
        state_config = self.get_state_config().get(current_state)
        
        if state_config:
            return self._state_machine.retry_count > state_config.max_retries
        return False
    
    def get_asr_hints(self) -> list[str]:
        """Get ASR hints for current state."""
        if not self._state_machine:
            return []
        return self._state_machine.get_asr_hints()
    
    # =========================================================================
    # Event Methods
    # =========================================================================
    
    async def _emit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> Optional[str]:
        """Emit an event to the event bus."""
        if not self.event_bus:
            return None
        
        event = VoiceAIEvent(
            event_type=event_type,
            agent_type=self.get_agent_type(),
            session_id=self._session_id,
            payload=payload,
            correlation_id=correlation_id,
            tenant_id=self.tenant_id,
        )
        
        try:
            await self.event_bus.publish(event)
            return event.event_id
        except Exception as e:
            logger.error(
                "event_publish_failed",
                event_type=event_type,
                error=str(e),
            )
            return None
    
    # =========================================================================
    # Checkpoint Methods
    # =========================================================================
    
    async def _create_checkpoint(self, reason: str = "periodic") -> Optional[str]:
        """Create a checkpoint of current state."""
        if not self.checkpoint_service or not self._context:
            return None
        
        self._checkpoint_count += 1
        
        snapshot = StateSnapshot(
            session_id=self._session_id,
            agent_type=self.get_agent_type(),
            tenant_id=self.tenant_id,
            current_state=self._context.current_state,
            previous_state=self._context.previous_state,
            retry_count=self._context.retry_count,
            checkpoint_reason=reason,
            checkpoint_number=self._checkpoint_count,
            state_entry_time=self._context.state_entry_time,
            session_start_time=self._context.created_at,
        )
        
        snapshot.set_messages(self._context.messages)
        snapshot.set_context(self._serialize_context())
        
        try:
            checkpoint_id = await self.checkpoint_service.save(snapshot)
            
            await self._emit_event(
                EventTypes.CHECKPOINT_CREATED,
                {"checkpoint_id": checkpoint_id, "reason": reason},
            )
            
            return checkpoint_id
        except Exception as e:
            logger.error(
                "checkpoint_save_failed",
                error=str(e),
                session_id=self._session_id[:8],
            )
            return None
    
    async def restore_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Restore agent state from a checkpoint."""
        if not self.checkpoint_service:
            return False
        
        try:
            if checkpoint_id:
                snapshot = await self.checkpoint_service.load(checkpoint_id)
            else:
                snapshot = await self.checkpoint_service.load_latest(self._session_id)
            
            if not snapshot:
                logger.warning(
                    "no_checkpoint_found",
                    session_id=self._session_id[:8],
                )
                return False
            
            self._context = self.create_context()
            self._context.session_id = snapshot.session_id
            self._context.tenant_id = snapshot.tenant_id
            self._context.current_state = snapshot.current_state
            self._context.previous_state = snapshot.previous_state
            self._context.retry_count = snapshot.retry_count
            self._context.messages = snapshot.get_messages()
            self._context.state_entry_time = snapshot.state_entry_time
            self._context.created_at = snapshot.session_start_time or datetime.now(timezone.utc)
            
            self._deserialize_context(snapshot.get_context())
            
            current_state = self._parse_state(snapshot.current_state)
            if current_state:
                self._state_machine = StateMachine(
                    agent_type=self.get_agent_type(),
                    state_config=self.get_state_config(),
                    initial_state=current_state,
                    session_id=self._session_id,
                )
            
            self._checkpoint_count = snapshot.checkpoint_number
            self._started = True
            self._ended = False
            
            logger.info(
                "checkpoint_restored",
                session_id=self._session_id[:8],
                checkpoint_id=snapshot.snapshot_id[:8],
                state=snapshot.current_state,
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "checkpoint_restore_failed",
                error=str(e),
                session_id=self._session_id[:8],
            )
            return False
    
    def _serialize_context(self) -> dict[str, Any]:
        """Serialize context for checkpointing. Override in subclass."""
        return {"metadata": self._context.metadata if self._context else {}}
    
    def _deserialize_context(self, data: dict[str, Any]) -> None:
        """Deserialize context from checkpoint. Override in subclass."""
        if self._context and "metadata" in data:
            self._context.metadata = data["metadata"]
    
    # =========================================================================
    # LLM Methods
    # =========================================================================
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        max_history: int = 10,
    ) -> str:
        """Generate LLM response."""
        if not self.llm:
            raise RuntimeError("No LLM configured")
        
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        if include_history and self._context:
            for msg in self._context.get_recent_messages(max_history):
                messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
        
        messages.append(LLMMessage(role="user", content=prompt))
        
        response = self.llm.generate(messages)
        return response.content
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def session_id(self) -> str:
        return self._session_id
    
    @property
    def context(self) -> Optional[TContext]:
        return self._context
    
    @property
    def is_started(self) -> bool:
        return self._started
    
    @property
    def is_ended(self) -> bool:
        return self._ended
    
    def get_state(self) -> dict[str, Any]:
        """Get full agent state as dictionary."""
        return {
            "session_id": self._session_id,
            "agent_type": self.get_agent_type(),
            "current_state": self._context.current_state if self._context else None,
            "started": self._started,
            "ended": self._ended,
            "duration_s": self._context.duration_s if self._context else 0,
            "message_count": len(self._context.messages) if self._context else 0,
            "checkpoint_count": self._checkpoint_count,
        }
