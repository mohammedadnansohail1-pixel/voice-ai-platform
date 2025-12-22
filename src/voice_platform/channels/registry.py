"""Channel registry."""
from ..core.registry import Registry
from .base import Channel

channel_registry = Registry[Channel]("Channel")
