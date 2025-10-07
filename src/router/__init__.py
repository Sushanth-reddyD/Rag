"""Router module initialization."""

from .models import RoutingDecision, RouterState
from .routing_logic import routing_decision

# Optional imports that require external dependencies
try:
    from .router_node import RouterNode, get_router
    from .orchestrator import LangGraphOrchestrator
    
    __all__ = [
        "RoutingDecision",
        "RouterState",
        "routing_decision",
        "RouterNode",
        "get_router",
        "LangGraphOrchestrator"
    ]
except ImportError:
    # If dependencies are not installed, only export core functionality
    __all__ = [
        "RoutingDecision",
        "RouterState",
        "routing_decision"
    ]
