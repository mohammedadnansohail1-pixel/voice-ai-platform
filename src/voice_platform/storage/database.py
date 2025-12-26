"""Database connection management for voice platform."""
import os
from typing import Optional
from contextlib import asynccontextmanager

from db_utils import DatabaseConnection, DatabaseConfig

from ..logging import get_logger
from ..core.exceptions import VoicePlatformError

logger = get_logger("storage.database")


class DatabaseError(VoicePlatformError):
    """Database connection or query error."""
    pass


# Global connection instance
_db_connection: Optional[DatabaseConnection] = None


def init_database(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    pool_size: int = 5,
) -> DatabaseConnection:
    """
    Initialize database connection.
    
    Uses environment variables as defaults:
    - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    """
    global _db_connection
    
    # Use env vars as defaults (production ready)
    config = DatabaseConfig(
        host=host or os.getenv("DB_HOST", "localhost"),
        port=port or int(os.getenv("DB_PORT", "5432")),
        user=user or os.getenv("DB_USER", "postgres"),
        password=password or os.getenv("DB_PASSWORD", ""),
        database=database or os.getenv("DB_NAME", "voice_ai"),
        pool_size=pool_size,
    )
    
    try:
        _db_connection = DatabaseConnection(config)
        logger.info(
            "database_initialized",
            host=config.host,
            database=config.database,
            pool_size=pool_size,
        )
        return _db_connection
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
        raise DatabaseError(f"Failed to initialize database: {e}") from e


def get_database() -> DatabaseConnection:
    """Get the database connection."""
    if _db_connection is None:
        logger.warning("database_not_initialized")
        raise DatabaseError("Database not initialized. Call init_database() first.")
    return _db_connection


@asynccontextmanager
async def get_session():
    """Get a database session with error handling."""
    db = get_database()
    try:
        async with db.session() as session:
            yield session
    except Exception as e:
        logger.error("database_session_error", error=str(e))
        raise DatabaseError(f"Database session error: {e}") from e
