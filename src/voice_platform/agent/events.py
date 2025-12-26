"""
Event Bus for Multi-Agent Coordination

Provides async pub/sub for agent communication with pluggable backends.
Currently supports Redis, designed for easy Kafka addition.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional
import uuid

from ..core.config import EventBusConfig
from ..logging import get_logger

logger = get_logger("agent.events")


# =============================================================================
# Event Definition
# =============================================================================

@dataclass
class VoiceAIEvent:
    """
    Event for multi-agent communication.
    
    Attributes:
        event_id: Unique ID for idempotency
        event_type: Event type (e.g., "appointment.booked", "insurance.verified")
        agent_type: Source agent (inbound, outbound, payer)
        session_id: Voice session this event belongs to
        payload: Event-specific data
        correlation_id: Links related events across agents
    """
    event_type: str
    agent_type: str
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    
    # Auto-generated fields
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    tenant_id: str = "default"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "tenant_id": self.tenant_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceAIEvent":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data["event_type"],
            agent_type=data["agent_type"],
            session_id=data["session_id"],
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=timestamp or datetime.now(timezone.utc),
            version=data.get("version", "1.0"),
            tenant_id=data.get("tenant_id", "default"),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "VoiceAIEvent":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Event Types Constants
# =============================================================================

class EventTypes:
    """Standard event types for the platform."""
    
    # Inbound Agent Events
    PATIENT_IDENTIFIED = "patient.identified"
    CONSENT_GIVEN = "consent.given"
    APPOINTMENT_REQUESTED = "appointment.requested"
    APPOINTMENT_BOOKED = "appointment.booked"
    CALL_TRANSFERRED = "call.transferred"
    INSURANCE_VERIFICATION_REQUESTED = "insurance.verification.requested"
    
    # Outbound Agent Events
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"
    REMINDER_DELIVERED = "reminder.delivered"
    APPOINTMENT_CONFIRMED = "appointment.confirmed"
    APPOINTMENT_RESCHEDULED = "appointment.rescheduled"
    APPOINTMENT_CANCELLED = "appointment.cancelled"
    VOICEMAIL_LEFT = "voicemail.left"
    
    # Payer Agent Events
    VERIFICATION_STARTED = "insurance.verification.started"
    HOLD_STARTED = "insurance.hold.started"
    HOLD_ENDED = "insurance.hold.ended"
    INSURANCE_VERIFIED = "insurance.verified"
    VERIFICATION_FAILED = "insurance.verification.failed"
    
    # System Events
    AGENT_STARTED = "agent.started"
    AGENT_ENDED = "agent.ended"
    CHECKPOINT_CREATED = "checkpoint.created"
    ERROR_OCCURRED = "error.occurred"


# =============================================================================
# Event Handler Type
# =============================================================================

EventHandler = Callable[[VoiceAIEvent], Coroutine[Any, Any, None]]


# =============================================================================
# Backend Interface
# =============================================================================

class EventBusBackend(ABC):
    """Abstract backend for event bus implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the backend."""
        pass
    
    @abstractmethod
    async def publish(self, channel: str, event: VoiceAIEvent) -> None:
        """Publish an event to a channel."""
        pass
    
    @abstractmethod
    async def subscribe(
        self, 
        channels: list[str], 
        handler: EventHandler,
    ) -> str:
        """
        Subscribe to channels with a handler.
        
        Returns:
            Subscription ID for later unsubscribe.
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a subscription."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        pass


# =============================================================================
# Redis Backend
# =============================================================================

class RedisEventBusBackend(EventBusBackend):
    """Redis pub/sub backend implementation."""
    
    def __init__(self, config: EventBusConfig) -> None:
        self.config = config
        self._redis: Optional[Any] = None  # redis.asyncio.Redis
        self._pubsub: Optional[Any] = None
        self._subscriptions: dict[str, asyncio.Task] = {}
        self._handlers: dict[str, list[EventHandler]] = {}
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("redis package required: pip install redis")
        
        self._redis = aioredis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_max_connections,
            socket_timeout=self.config.redis_socket_timeout,
        )
        
        # Test connection
        await self._redis.ping()
        self._connected = True
        logger.info("redis_connected", url=self.config.redis_url)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        # Cancel all subscription tasks
        for task in self._subscriptions.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._subscriptions.clear()
        self._handlers.clear()
        
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None
        
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        self._connected = False
        logger.info("redis_disconnected")
    
    async def publish(self, channel: str, event: VoiceAIEvent) -> None:
        """Publish event to Redis channel."""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        message = event.to_json()
        await self._redis.publish(channel, message)
        
        logger.debug(
            "event_published",
            channel=channel,
            event_type=event.event_type,
            event_id=event.event_id[:8],
        )
    
    async def subscribe(
        self,
        channels: list[str],
        handler: EventHandler,
    ) -> str:
        """Subscribe to Redis channels."""
        if not self._redis:
            raise RuntimeError("Redis not connected")
        
        subscription_id = str(uuid.uuid4())
        
        # Track handler for each channel
        for channel in channels:
            if channel not in self._handlers:
                self._handlers[channel] = []
            self._handlers[channel].append(handler)
        
        # Create pubsub if needed
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()
        
        # Subscribe to channels
        await self._pubsub.subscribe(*channels)
        
        # Start listener task if not already running
        if "listener" not in self._subscriptions:
            task = asyncio.create_task(self._listen())
            self._subscriptions["listener"] = task
        
        logger.info(
            "subscribed",
            subscription_id=subscription_id[:8],
            channels=channels,
        )
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe (currently unsubscribes all - simplified implementation)."""
        if self._pubsub:
            await self._pubsub.unsubscribe()
        self._handlers.clear()
        logger.info("unsubscribed", subscription_id=subscription_id[:8])
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def _listen(self) -> None:
        """Listen for messages and dispatch to handlers."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")
                    
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    
                    try:
                        event = VoiceAIEvent.from_json(data)
                    except Exception as e:
                        logger.error("event_parse_error", error=str(e), data=data[:100])
                        continue
                    
                    # Dispatch to handlers
                    handlers = self._handlers.get(channel, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(
                                "handler_error",
                                error=str(e),
                                event_type=event.event_type,
                                handler=handler.__name__,
                            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("listener_error", error=str(e))


# =============================================================================
# Main Event Bus
# =============================================================================

class EventBus:
    """
    Main event bus for multi-agent coordination.
    
    Usage:
        config = EventBusConfig()
        bus = EventBus(config)
        await bus.start()
        
        # Publish
        event = VoiceAIEvent(
            event_type=EventTypes.APPOINTMENT_BOOKED,
            agent_type="inbound",
            session_id="abc123",
            payload={"appointment_id": "xyz"}
        )
        await bus.publish(event)
        
        # Subscribe
        async def handle_booking(event: VoiceAIEvent):
            print(f"Booking: {event.payload}")
        
        await bus.subscribe([EventTypes.APPOINTMENT_BOOKED], handle_booking)
        
        # Cleanup
        await bus.stop()
    """
    
    def __init__(self, config: EventBusConfig) -> None:
        self.config = config
        self._backend: Optional[EventBusBackend] = None
        self._started = False
        
        # Track processed event IDs for idempotency
        self._processed_events: set[str] = set()
        self._max_processed_cache = 10000
    
    async def start(self) -> None:
        """Initialize and connect the event bus."""
        if self._started:
            return
        
        # Select backend based on config
        if self.config.backend == "redis":
            self._backend = RedisEventBusBackend(self.config)
        elif self.config.backend == "kafka":
            raise NotImplementedError("Kafka backend not yet implemented")
        else:
            raise ValueError(f"Unknown event bus backend: {self.config.backend}")
        
        await self._backend.connect()
        self._started = True
        logger.info("event_bus_started", backend=self.config.backend)
    
    async def stop(self) -> None:
        """Stop and disconnect the event bus."""
        if not self._started:
            return
        
        if self._backend:
            await self._backend.disconnect()
            self._backend = None
        
        self._started = False
        self._processed_events.clear()
        logger.info("event_bus_stopped")
    
    def _get_channel(self, event_type: str, tenant_id: str = "default") -> str:
        """Build channel name from prefix, tenant, and event type."""
        return f"{self.config.channel_prefix}:{tenant_id}:{event_type}"
    
    async def publish(self, event: VoiceAIEvent) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: The event to publish
        """
        if not self._backend:
            raise RuntimeError("Event bus not started")
        
        channel = self._get_channel(event.event_type, event.tenant_id)
        await self._backend.publish(channel, event)
        
        logger.info(
            "event_published",
            event_type=event.event_type,
            session_id=event.session_id[:8] if event.session_id else "none",
            event_id=event.event_id[:8],
        )
    
    async def subscribe(
        self,
        event_types: list[str],
        handler: EventHandler,
        tenant_id: str = "default",
        idempotent: bool = True,
    ) -> str:
        """
        Subscribe to event types.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Async function to handle events
            tenant_id: Tenant to subscribe for
            idempotent: If True, skip already-processed events
            
        Returns:
            Subscription ID
        """
        if not self._backend:
            raise RuntimeError("Event bus not started")
        
        channels = [self._get_channel(et, tenant_id) for et in event_types]
        
        # Wrap handler with idempotency check if requested
        if idempotent:
            original_handler = handler
            
            async def idempotent_handler(event: VoiceAIEvent) -> None:
                if event.event_id in self._processed_events:
                    logger.debug(
                        "event_skipped_duplicate",
                        event_id=event.event_id[:8],
                    )
                    return
                
                # Add to processed set
                self._processed_events.add(event.event_id)
                
                # Trim cache if too large
                if len(self._processed_events) > self._max_processed_cache:
                    # Remove oldest (arbitrary since set, but prevents unbounded growth)
                    to_remove = list(self._processed_events)[:1000]
                    for eid in to_remove:
                        self._processed_events.discard(eid)
                
                await original_handler(event)
            
            handler = idempotent_handler
        
        subscription_id = await self._backend.subscribe(channels, handler)
        
        logger.info(
            "subscribed",
            event_types=event_types,
            subscription_id=subscription_id[:8],
        )
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a subscription."""
        if not self._backend:
            return
        
        await self._backend.unsubscribe(subscription_id)
    
    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._started and self._backend is not None and self._backend.is_connected


# =============================================================================
# Factory Function
# =============================================================================

async def create_event_bus(config: EventBusConfig) -> EventBus:
    """Create and start an event bus."""
    bus = EventBus(config)
    await bus.start()
    return bus
