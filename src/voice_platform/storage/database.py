"""Database connection management for voice platform."""
from typing import Optional
from contextlib import asynccontextmanager

from db_utils import DatabaseConnection, DatabaseConfig

from ..logging import get_logger

logger = get_logger("storage.database")

# Global connection instance
_db_connection: Optional[DatabaseConnection] = None


def init_database(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    database: str = "voice_ai",
    pool_size: int = 5,
) -> DatabaseConnection:
    """Initialize database connection."""
    global _db_connection
    
    config = DatabaseConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        pool_size=pool_size,
    )
    
    _db_connection = DatabaseConnection(config)
    logger.info("database_initialized", host=host, database=database)
    return _db_connection


def get_database() -> DatabaseConnection:
    """Get the database connection."""
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_connection


@asynccontextmanager
async def get_session():
    """Get a database session."""
    db = get_database()
    async with db.session() as session:
        yield session
