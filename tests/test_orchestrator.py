"""Test suite for orchestrator functionality."""

import pytest
from src.router.orchestrator import LangGraphOrchestrator


class TestLangGraphOrchestrator:
    """Test cases for LangGraphOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance for testing."""
        return LangGraphOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.router is not None
        assert orchestrator.graph is not None
    
    def test_route_retrieval_query(self, orchestrator):
        """Test routing a retrieval query through orchestrator."""
        result = orchestrator.route_query("What is your return policy?")
        assert result["routing_decision"] == "retrieval"
        assert result["user_input"] == "What is your return policy?"
        assert "response" in result
    
    def test_route_conversational_query(self, orchestrator):
        """Test routing a conversational query through orchestrator."""
        result = orchestrator.route_query("Hello, how are you?")
        assert result["routing_decision"] == "conversational"
        assert "response" in result
    
    def test_route_api_call_query(self, orchestrator):
        """Test routing an API call query through orchestrator."""
        result = orchestrator.route_query("What's the weather today?")
        assert result["routing_decision"] == "api_call"
        assert "response" in result
    
    def test_route_complaint_query(self, orchestrator):
        """Test routing a complaint query through orchestrator."""
        result = orchestrator.route_query("My product is broken!")
        assert result["routing_decision"] == "complaint"
        assert "response" in result
    
    def test_orchestrator_state_flow(self, orchestrator):
        """Test complete state flow through orchestrator."""
        result = orchestrator.route_query("How do I reset my password?")
        
        # Verify all state fields are present
        assert "user_input" in result
        assert "routing_decision" in result
        assert "reasoning" in result
        assert "confidence" in result
        assert "response" in result
        
        # Verify routing decision
        assert result["routing_decision"] == "retrieval"


class TestOrchestratorEdgeCases:
    """Test edge cases for orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance for testing."""
        return LangGraphOrchestrator()
    
    def test_empty_query(self, orchestrator):
        """Test handling of empty query."""
        result = orchestrator.route_query("")
        assert result["routing_decision"] in ["conversational", "retrieval", "api_call", "complaint"]
        assert "response" in result
    
    def test_very_long_query(self, orchestrator):
        """Test handling of very long query."""
        long_query = "What is your return policy? " * 50
        result = orchestrator.route_query(long_query)
        assert result["routing_decision"] is not None
        assert "response" in result
    
    def test_special_characters_query(self, orchestrator):
        """Test handling of query with special characters."""
        result = orchestrator.route_query("What's your policy on $$ refunds???")
        assert result["routing_decision"] is not None
        assert "response" in result
    
    def test_multiple_categories_query(self, orchestrator):
        """Test query that could fit multiple categories."""
        result = orchestrator.route_query("Hello! What's your return policy?")
        # Should prioritize based on routing logic
        assert result["routing_decision"] in ["retrieval", "conversational"]
        assert "response" in result
