"""Standalone tests for routing logic without full dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_routing_decision_function():
    """Test the routing decision function."""
    # Import only the routing logic module
    from src.router.routing_logic import routing_decision
    
    # Test valid decisions
    assert routing_decision({"routing_decision": "complaint"}) == "complaint"
    assert routing_decision({"routing_decision": "api_call"}) == "api_call"
    assert routing_decision({"routing_decision": "retrieval"}) == "retrieval"
    assert routing_decision({"routing_decision": "conversational"}) == "conversational"
    
    # Test invalid decision defaults to conversational
    assert routing_decision({"routing_decision": "invalid"}) == "conversational"
    
    # Test missing decision defaults to conversational
    assert routing_decision({}) == "conversational"
    
    print("✅ All routing_decision tests passed!")


def test_routing_decision_model():
    """Test the RoutingDecision pydantic model."""
    from src.router.models import RoutingDecision
    
    # Test valid model creation
    decision = RoutingDecision(
        category="complaint",
        reasoning="Test reasoning",
        confidence="high"
    )
    
    assert decision.category == "complaint"
    assert decision.confidence == "high"
    assert decision.reasoning == "Test reasoning"
    
    # Test all valid categories
    for category in ["complaint", "api_call", "retrieval", "conversational"]:
        d = RoutingDecision(
            category=category,
            reasoning="Test",
            confidence="medium"
        )
        assert d.category == category
    
    # Test all valid confidence levels
    for confidence in ["high", "medium", "low"]:
        d = RoutingDecision(
            category="complaint",
            reasoning="Test",
            confidence=confidence
        )
        assert d.confidence == confidence
    
    print("✅ All RoutingDecision model tests passed!")


def test_router_state_model():
    """Test the RouterState pydantic model."""
    from src.router.models import RouterState
    
    state = RouterState(
        user_input="Test query",
        routing_decision="retrieval",
        reasoning="Test reasoning",
        confidence="high",
        response="Test response"
    )
    
    assert state.user_input == "Test query"
    assert state.routing_decision == "retrieval"
    assert state.reasoning == "Test reasoning"
    assert state.confidence == "high"
    assert state.response == "Test response"
    
    print("✅ All RouterState model tests passed!")


if __name__ == "__main__":
    print("="*80)
    print("🧪 Running Standalone Unit Tests")
    print("="*80)
    print()
    
    try:
        test_routing_decision_function()
        test_routing_decision_model()
        test_router_state_model()
        
        print()
        print("="*80)
        print("✅ All tests passed successfully!")
        print("="*80)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ Tests failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
