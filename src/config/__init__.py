"""Init file for config module."""
from .settings import (
    MODEL_NAME,
    DEVICE,
    TEMPERATURE,
    MAX_TOKENS,
    ROUTING_CATEGORIES,
    ROUTING_PRIORITY,
    USE_FINE_TUNED,
    FINE_TUNED_MODEL_PATH
)

# Import model_config for generation models
try:
    from . import model_config
except ImportError:
    model_config = None

__all__ = [
    "MODEL_NAME",
    "DEVICE",
    "TEMPERATURE",
    "MAX_TOKENS",
    "ROUTING_CATEGORIES",
    "ROUTING_PRIORITY",
    "USE_FINE_TUNED",
    "FINE_TUNED_MODEL_PATH",
    "model_config"
]
