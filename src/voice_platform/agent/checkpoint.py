"""
Checkpoint Service for Agent Crash Recovery

Persists agent state for recovery during long-running calls (e.g., payer holds).
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import uuid

from ..core.config import CheckpointConfig
from ..logging import get_logger

logger = get_logger("agent.checkpoint")


# =============================================================================
# State Snapshot
# =============================================================================

@dataclass
class StateSnapshot:
    """
    Serializable snapshot of agent state for recovery.
    
    Contains everything needed to resume an agent after crash.
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Identity
    session_id: str = ""
    agent_type: str = ""  # inbound, outbound, payer
    tenant_id: str = "default"
    
    # State machine
    current_state: str = ""
    previous_state: Optional[str] = None
    retry_count: int = 0
    
    # Context (serialized)
    context_json: str = "{}"
    
    # Conversation
    messages_json: str = "[]"
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state_entry_time: Optional[datetime] = None
    session_start_time: Optional[datetime] = None
    
    # Metadata
    metadata_json: str = "{}"
    
    # Recovery info
    checkpoint_reason: str = "periodic"  # periodic, state_change, manual
    checkpoint_number: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "tenant_id": self.tenant_id,
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "retry_count": self.retry_count,
            "context_json": self.context_json,
            "messages_json": self.messages_json,
            "created_at": self.created_at.isoformat(),
            "state_entry_time": self.state_entry_time.isoformat() if self.state_entry_time else None,
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "metadata_json": self.metadata_json,
            "checkpoint_reason": self.checkpoint_reason,
            "checkpoint_number": self.checkpoint_number,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSnapshot":
        """Create from dictionary."""
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)
        
        return cls(
            snapshot_id=data.get("snapshot_id", str(uuid.uuid4())),
            session_id=data.get("session_id", ""),
            agent_type=data.get("agent_type", ""),
            tenant_id=data.get("tenant_id", "default"),
            current_state=data.get("current_state", ""),
            previous_state=data.get("previous_state"),
            retry_count=data.get("retry_count", 0),
            context_json=data.get("context_json", "{}"),
            messages_json=data.get("messages_json", "[]"),
            created_at=parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
            state_entry_time=parse_dt(data.get("state_entry_time")),
            session_start_time=parse_dt(data.get("session_start_time")),
            metadata_json=data.get("metadata_json", "{}"),
            checkpoint_reason=data.get("checkpoint_reason", "periodic"),
            checkpoint_number=data.get("checkpoint_number", 0),
        )
    
    def get_context(self) -> dict[str, Any]:
        """Deserialize context."""
        return json.loads(self.context_json)
    
    def set_context(self, context: dict[str, Any]) -> None:
        """Serialize context."""
        self.context_json = json.dumps(context, default=str)
    
    def get_messages(self) -> list[dict]:
        """Deserialize messages."""
        return json.loads(self.messages_json)
    
    def set_messages(self, messages: list[dict]) -> None:
        """Serialize messages."""
        self.messages_json = json.dumps(messages, default=str)
    
    def get_metadata(self) -> dict[str, Any]:
        """Deserialize metadata."""
        return json.loads(self.metadata_json)
    
    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Serialize metadata."""
        self.metadata_json = json.dumps(metadata, default=str)


# =============================================================================
# Backend Interface
# =============================================================================

class CheckpointBackend(ABC):
    """Abstract backend for checkpoint storage."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass
    
    @abstractmethod
    async def save(self, snapshot: StateSnapshot) -> str:
        """Save snapshot, return snapshot_id."""
        pass
    
    @abstractmethod
    async def load(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load snapshot by ID."""
        pass
    
    @abstractmethod
    async def load_latest(self, session_id: str) -> Optional[StateSnapshot]:
        """Load most recent snapshot for session."""
        pass
    
    @abstractmethod
    async def list_by_session(self, session_id: str) -> list[StateSnapshot]:
        """List all snapshots for session."""
        pass
    
    @abstractmethod
    async def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        pass
    
    @abstractmethod
    async def cleanup(self, older_than: datetime) -> int:
        """Delete snapshots older than timestamp, return count deleted."""
        pass
    
    @abstractmethod
    async def ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        pass


# =============================================================================
# PostgreSQL Backend
# =============================================================================

class PostgreSQLCheckpointBackend(CheckpointBackend):
    """PostgreSQL checkpoint storage backend."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_checkpoints (
        snapshot_id VARCHAR(36) PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        agent_type VARCHAR(50) NOT NULL,
        tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
        current_state VARCHAR(100) NOT NULL,
        previous_state VARCHAR(100),
        retry_count INTEGER NOT NULL DEFAULT 0,
        context_json TEXT NOT NULL DEFAULT '{}',
        messages_json TEXT NOT NULL DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        state_entry_time TIMESTAMPTZ,
        session_start_time TIMESTAMPTZ,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        checkpoint_reason VARCHAR(50) NOT NULL DEFAULT 'periodic',
        checkpoint_number INTEGER NOT NULL DEFAULT 0
    );
    
    CREATE INDEX IF NOT EXISTS idx_checkpoints_session 
        ON agent_checkpoints(session_id, created_at DESC);
    
    CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
        ON agent_checkpoints(created_at);
    
    CREATE INDEX IF NOT EXISTS idx_checkpoints_tenant_session 
        ON agent_checkpoints(tenant_id, session_id);
    """
    
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self._pool: Optional[Any] = None  # asyncpg.Pool
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg package required: pip install asyncpg")
        
        self._pool = await asyncpg.create_pool(
            self.config.dsn,
            min_size=2,
            max_size=10,
        )
        logger.info("postgresql_connected", dsn=self._mask_dsn(self.config.dsn))
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        logger.info("postgresql_disconnected")
    
    def _mask_dsn(self, dsn: str) -> str:
        """Mask password in DSN for logging."""
        import re
        return re.sub(r':([^:@]+)@', ':****@', dsn)
    
    async def ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            await conn.execute(self.SCHEMA)
        logger.info("schema_ensured", table="agent_checkpoints")
    
    async def save(self, snapshot: StateSnapshot) -> str:
        """Save snapshot to PostgreSQL."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_checkpoints (
                    snapshot_id, session_id, agent_type, tenant_id,
                    current_state, previous_state, retry_count,
                    context_json, messages_json, created_at,
                    state_entry_time, session_start_time,
                    metadata_json, checkpoint_reason, checkpoint_number
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (snapshot_id) DO UPDATE SET
                    current_state = EXCLUDED.current_state,
                    previous_state = EXCLUDED.previous_state,
                    retry_count = EXCLUDED.retry_count,
                    context_json = EXCLUDED.context_json,
                    messages_json = EXCLUDED.messages_json,
                    metadata_json = EXCLUDED.metadata_json,
                    checkpoint_reason = EXCLUDED.checkpoint_reason,
                    checkpoint_number = EXCLUDED.checkpoint_number
                """,
                snapshot.snapshot_id,
                snapshot.session_id,
                snapshot.agent_type,
                snapshot.tenant_id,
                snapshot.current_state,
                snapshot.previous_state,
                snapshot.retry_count,
                snapshot.context_json,
                snapshot.messages_json,
                snapshot.created_at,
                snapshot.state_entry_time,
                snapshot.session_start_time,
                snapshot.metadata_json,
                snapshot.checkpoint_reason,
                snapshot.checkpoint_number,
            )
        
        logger.debug(
            "checkpoint_saved",
            snapshot_id=snapshot.snapshot_id[:8],
            session_id=snapshot.session_id[:8],
            state=snapshot.current_state,
        )
        
        return snapshot.snapshot_id
    
    async def load(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load snapshot by ID."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_checkpoints WHERE snapshot_id = $1",
                snapshot_id,
            )
        
        if not row:
            return None
        
        return self._row_to_snapshot(row)
    
    async def load_latest(self, session_id: str) -> Optional[StateSnapshot]:
        """Load most recent snapshot for session."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM agent_checkpoints 
                WHERE session_id = $1 
                ORDER BY created_at DESC 
                LIMIT 1
                """,
                session_id,
            )
        
        if not row:
            return None
        
        return self._row_to_snapshot(row)
    
    async def list_by_session(self, session_id: str) -> list[StateSnapshot]:
        """List all snapshots for session."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM agent_checkpoints 
                WHERE session_id = $1 
                ORDER BY created_at DESC
                """,
                session_id,
            )
        
        return [self._row_to_snapshot(row) for row in rows]
    
    async def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_checkpoints WHERE snapshot_id = $1",
                snapshot_id,
            )
        
        deleted = result.split()[-1] != "0"
        if deleted:
            logger.debug("checkpoint_deleted", snapshot_id=snapshot_id[:8])
        return deleted
    
    async def cleanup(self, older_than: datetime) -> int:
        """Delete old snapshots."""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_checkpoints WHERE created_at < $1",
                older_than,
            )
        
        # Parse "DELETE N" result
        count = int(result.split()[-1])
        if count > 0:
            logger.info("checkpoints_cleaned", count=count, older_than=older_than.isoformat())
        return count
    
    def _row_to_snapshot(self, row) -> StateSnapshot:
        """Convert database row to StateSnapshot."""
        return StateSnapshot(
            snapshot_id=row["snapshot_id"],
            session_id=row["session_id"],
            agent_type=row["agent_type"],
            tenant_id=row["tenant_id"],
            current_state=row["current_state"],
            previous_state=row["previous_state"],
            retry_count=row["retry_count"],
            context_json=row["context_json"],
            messages_json=row["messages_json"],
            created_at=row["created_at"],
            state_entry_time=row["state_entry_time"],
            session_start_time=row["session_start_time"],
            metadata_json=row["metadata_json"],
            checkpoint_reason=row["checkpoint_reason"],
            checkpoint_number=row["checkpoint_number"],
        )


# =============================================================================
# Checkpoint Service
# =============================================================================

class CheckpointService:
    """
    Main checkpoint service for agent crash recovery.
    
    Usage:
        config = CheckpointConfig()
        service = CheckpointService(config)
        await service.start()
        
        # Save checkpoint
        snapshot = StateSnapshot(
            session_id="abc",
            agent_type="payer",
            current_state="WAITING_HOLD",
        )
        snapshot_id = await service.save(snapshot)
        
        # Recover from crash
        latest = await service.load_latest("abc")
        if latest:
            # Resume from latest.current_state
            pass
        
        await service.stop()
    """
    
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self._backend: Optional[CheckpointBackend] = None
        self._started = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the checkpoint service."""
        if self._started:
            return
        
        # Select backend
        if self.config.backend == "postgresql":
            self._backend = PostgreSQLCheckpointBackend(self.config)
        else:
            raise ValueError(f"Unknown checkpoint backend: {self.config.backend}")
        
        await self._backend.connect()
        await self._backend.ensure_schema()
        
        # Start periodic cleanup
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._started = True
        logger.info("checkpoint_service_started", backend=self.config.backend)
    
    async def stop(self) -> None:
        """Stop the checkpoint service."""
        if not self._started:
            return
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        if self._backend:
            await self._backend.disconnect()
            self._backend = None
        
        self._started = False
        logger.info("checkpoint_service_stopped")
    
    async def save(self, snapshot: StateSnapshot) -> str:
        """Save a checkpoint."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        return await self._backend.save(snapshot)
    
    async def load(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load checkpoint by ID."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        return await self._backend.load(snapshot_id)
    
    async def load_latest(self, session_id: str) -> Optional[StateSnapshot]:
        """Load most recent checkpoint for session."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        return await self._backend.load_latest(session_id)
    
    async def list_by_session(self, session_id: str) -> list[StateSnapshot]:
        """List all checkpoints for session."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        return await self._backend.list_by_session(session_id)
    
    async def delete(self, snapshot_id: str) -> bool:
        """Delete a checkpoint."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        return await self._backend.delete(snapshot_id)
    
    async def cleanup_old(self) -> int:
        """Manually trigger cleanup of old checkpoints."""
        if not self._backend:
            raise RuntimeError("Checkpoint service not started")
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.retention_hours)
        return await self._backend.cleanup(cutoff)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old checkpoints."""
        try:
            while True:
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
                try:
                    count = await self.cleanup_old()
                    if count > 0:
                        logger.info("periodic_cleanup_complete", deleted=count)
                except Exception as e:
                    logger.error("cleanup_error", error=str(e))
        
        except asyncio.CancelledError:
            pass


# =============================================================================
# Factory Function
# =============================================================================

async def create_checkpoint_service(config: CheckpointConfig) -> CheckpointService:
    """Create and start a checkpoint service."""
    service = CheckpointService(config)
    await service.start()
    return service
