"""Structured logging setup."""
import logging
import sys
import structlog
from typing import Optional

from ..core.config import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure structured logging."""
    if config is None:
        config = LoggingConfig()
    
    # Convert string level to int
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Processors for all log entries
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)
