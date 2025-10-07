"""Test suite for router functionality."""

import pytest
from src.router.router_node import RouterNode
from src.router.routing_logic import routing_decision
from src.router.models import RoutingDecision


class TestRouterNode:
    """Test cases for RouterNode."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance for testing."""
        return RouterNode()
    
    # Retrieval Test Cases
    def test_return_policy_query(self, router):
        """Test routing for return policy query."""
        state = {"user_input": "What is your return policy?"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
        assert "user_input" in result
    
    def test_password_reset_query(self, router):
        """Test routing for password reset query."""
        state = {"user_input": "How do I reset my password?"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    def test_warranty_info_query(self, router):
        """Test routing for warranty information query."""
        state = {"user_input": "Where can I find the warranty information?"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    def test_shipping_policy_query(self, router):
        """Test routing for shipping policy query."""
        state = {"user_input": "What are your shipping policies?"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    def test_privacy_policy_query(self, router):
        """Test routing for privacy policy query."""
        state = {"user_input": "Tell me about your company's privacy policy"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    def test_refund_request_query(self, router):
        """Test routing for refund request query."""
        state = {"user_input": "How do I submit a refund request?"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    # Conversational Test Cases
    def test_greeting_query(self, router):
        """Test routing for greeting."""
        state = {"user_input": "Hello, how are you?"}
        result = router.route(state)
        assert result["routing_decision"] == "conversational"
    
    def test_thanks_query(self, router):
        """Test routing for thanks."""
        state = {"user_input": "Thanks for your help!"}
        result = router.route(state)
        assert result["routing_decision"] == "conversational"
    
    # API Call Test Cases
    def test_weather_query(self, router):
        """Test routing for weather query."""
        state = {"user_input": "What's the weather in London?"}
        result = router.route(state)
        assert result["routing_decision"] == "api_call"
    
    def test_order_tracking_query(self, router):
        """Test routing for order tracking."""
        state = {"user_input": "Track my order #12345"}
        result = router.route(state)
        assert result["routing_decision"] == "api_call"
    
    # Complaint Test Cases
    def test_broken_product_complaint(self, router):
        """Test routing for broken product complaint."""
        state = {"user_input": "My product arrived broken!"}
        result = router.route(state)
        assert result["routing_decision"] == "complaint"
    
    def test_delayed_order_complaint(self, router):
        """Test routing for delayed order complaint."""
        state = {"user_input": "I've been waiting for 2 weeks, this is unacceptable!"}
        result = router.route(state)
        assert result["routing_decision"] == "complaint"
    
    # Edge Cases
    def test_edge_case_complaint_with_policy(self, router):
        """Test edge case: complaint mentioned with policy."""
        state = {"user_input": "Can you help me understand your return policy? Mine is defective."}
        result = router.route(state)
        # Should prioritize complaint due to "defective"
        assert result["routing_decision"] == "complaint"
    
    def test_edge_case_documentation_request(self, router):
        """Test edge case: documentation request without problem."""
        state = {"user_input": "I need the documentation for returns"}
        result = router.route(state)
        assert result["routing_decision"] == "retrieval"
    
    def test_empty_query(self, router):
        """Test routing for empty query."""
        state = {"user_input": ""}
        result = router.route(state)
        # Should default to conversational
        assert result["routing_decision"] == "conversational"


class TestRoutingLogic:
    """Test cases for routing logic."""
    
    def test_valid_complaint_decision(self):
        """Test valid complaint routing decision."""
        state = {"routing_decision": "complaint"}
        result = routing_decision(state)
        assert result == "complaint"
    
    def test_valid_api_call_decision(self):
        """Test valid api_call routing decision."""
        state = {"routing_decision": "api_call"}
        result = routing_decision(state)
        assert result == "api_call"
    
    def test_valid_retrieval_decision(self):
        """Test valid retrieval routing decision."""
        state = {"routing_decision": "retrieval"}
        result = routing_decision(state)
        assert result == "retrieval"
    
    def test_valid_conversational_decision(self):
        """Test valid conversational routing decision."""
        state = {"routing_decision": "conversational"}
        result = routing_decision(state)
        assert result == "conversational"
    
    def test_invalid_decision_defaults_to_conversational(self):
        """Test invalid decision defaults to conversational."""
        state = {"routing_decision": "invalid_category"}
        result = routing_decision(state)
        assert result == "conversational"
    
    def test_missing_decision_defaults_to_conversational(self):
        """Test missing decision defaults to conversational."""
        state = {}
        result = routing_decision(state)
        assert result == "conversational"


class TestRoutingDecisionModel:
    """Test cases for RoutingDecision model."""
    
    def test_valid_routing_decision(self):
        """Test creating a valid routing decision."""
        decision = RoutingDecision(
            category="complaint",
            reasoning="Customer expressed dissatisfaction",
            confidence="high"
        )
        assert decision.category == "complaint"
        assert decision.confidence == "high"
    
    def test_invalid_category_raises_error(self):
        """Test invalid category raises validation error."""
        with pytest.raises(Exception):
            RoutingDecision(
                category="invalid",
                reasoning="Test",
                confidence="high"
            )
    
    def test_invalid_confidence_raises_error(self):
        """Test invalid confidence raises validation error."""
        with pytest.raises(Exception):
            RoutingDecision(
                category="complaint",
                reasoning="Test",
                confidence="invalid"
            )
