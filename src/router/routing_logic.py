"""Conditional routing logic for LangGraph."""

from typing import Dict, Any, Literal


def routing_decision(state: Dict[str, Any]) -> Literal["complaint", "api_call", "retrieval", "conversational"]:
    """
    Conditional edge function that determines which node to route to.
    
    Args:
        state: The current state containing routing_decision
        
    Returns:
        The category to route to
    """
    decision = state.get("routing_decision", "conversational")
    
    # Validate decision
    valid_categories = ["complaint", "api_call", "retrieval", "conversational"]
    if decision not in valid_categories:
        print(f"⚠️ Invalid routing decision: {decision}, defaulting to conversational")
        return "conversational"
    
    return decision
