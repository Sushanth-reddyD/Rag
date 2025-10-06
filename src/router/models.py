"""Routing decision model with structured output."""

from typing import Literal
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Structured output for routing decisions."""
    
    category: Literal["complaint", "api_call", "retrieval", "conversational"] = Field(
        description="The category to route the query to"
    )
    reasoning: str = Field(
        description="Brief explanation for the routing decision",
        max_length=200
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the routing decision"
    )


class RouterState(BaseModel):
    """State model for the router graph."""
    
    user_input: str = Field(description="The user's query")
    routing_decision: str = Field(default="", description="The routing category")
    reasoning: str = Field(default="", description="Reasoning for the decision")
    confidence: str = Field(default="medium", description="Confidence level")
    response: str = Field(default="", description="Agent response")
