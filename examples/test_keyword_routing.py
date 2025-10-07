"""Quick test script for keyword-based routing (no model download required)."""

import sys
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


def keyword_based_routing(user_input: str) -> RoutingDecision:
    """Fallback keyword-based routing."""
    user_input_lower = user_input.lower()
    
    # Complaint keywords (highest priority) - focus on emotional/problem language
    complaint_keywords = [
        "broken", "defective", "unacceptable", "angry", "frustrated",
        "complaint", "complain", "unhappy", "disappointed", "terrible",
        "awful", "worst", "hate", "never again", "damaged"
    ]
    
    # API call keywords
    api_keywords = [
        "weather", "stock", "current", "status", "track", "tracking",
        "order status", "delivery status", "what's the", "check status"
    ]
    
    # Retrieval keywords - includes procedural questions
    retrieval_keywords = [
        "policy", "policies", "documentation", "how do i", "where can i find",
        "what is your", "what are your", "procedure", "terms", "privacy", "warranty",
        "about us", "faq", "guide", "manual", "instructions", "shipping", "submit"
    ]
    
    # Conversational keywords
    conversational_keywords = [
        "hello", "hi", "hey", "good morning", "good evening",
        "thanks", "thank you", "bye", "goodbye", "joke"
    ]
    
    # Check in priority order
    if any(keyword in user_input_lower for keyword in complaint_keywords):
        return RoutingDecision(
            category="complaint",
            reasoning="Detected complaint-related keywords",
            confidence="high"
        )
    
    if any(keyword in user_input_lower for keyword in api_keywords):
        return RoutingDecision(
            category="api_call",
            reasoning="Detected real-time data request keywords",
            confidence="high"
        )
    
    if any(keyword in user_input_lower for keyword in retrieval_keywords):
        return RoutingDecision(
            category="retrieval",
            reasoning="Detected documentation/policy keywords",
            confidence="high"
        )
    
    if any(keyword in user_input_lower for keyword in conversational_keywords):
        return RoutingDecision(
            category="conversational",
            reasoning="Detected conversational keywords",
            confidence="high"
        )
    
    # Default to conversational
    return RoutingDecision(
        category="conversational",
        reasoning="No specific keywords detected, defaulting to conversational",
        confidence="low"
    )


def test_keyword_routing():
    """Test keyword-based routing without LLM."""
    
    print("="*80)
    print("🧪 Testing Keyword-Based Routing (No Model Required)")
    print("="*80)
    
    print("\n🔍 Testing keyword-based routing...\n")
    
    print("\n🔍 Testing keyword-based routing...\n")
    
    # Test cases
    test_queries = [
        # Retrieval queries
        ("What is your return policy?", "retrieval"),
        ("How do I reset my password?", "retrieval"),
        ("Where can I find the warranty information?", "retrieval"),
        ("What are your shipping policies?", "retrieval"),
        ("Tell me about your company's privacy policy", "retrieval"),
        ("How do I submit a refund request?", "retrieval"),
        
        # Conversational queries
        ("Hello, how are you?", "conversational"),
        ("Thanks for your help!", "conversational"),
        ("Good morning!", "conversational"),
        ("Tell me a joke", "conversational"),
        
        # API call queries
        ("What's the weather in London?", "api_call"),
        ("Track my order #12345", "api_call"),
        ("Check the current stock price", "api_call"),
        ("What's the order status?", "api_call"),
        
        # Complaint queries
        ("My product arrived broken!", "complaint"),
        ("I've been waiting for 2 weeks, this is unacceptable!", "complaint"),
        ("This is terrible service", "complaint"),
        ("I'm very unhappy with this", "complaint"),
        
        # Edge cases
        ("Can you help me understand your return policy? Mine is defective.", "complaint"),
        ("I need the documentation for returns", "retrieval"),
    ]
    
    correct_count = 0
    total_count = len(test_queries)
    
    for query, expected_category in test_queries:
        decision = keyword_based_routing(query)
        actual_category = decision.category
        
        is_correct = actual_category == expected_category
        
        if is_correct:
            correct_count += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} Query: {query[:50]:<50} | Expected: {expected_category:<15} | Actual: {actual_category:<15} | Confidence: {decision.confidence}")
    
    # Calculate accuracy
    accuracy = (correct_count / total_count) * 100
    
    print("\n" + "="*80)
    print(f"📊 RESULTS: {correct_count}/{total_count} correct")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 85:
        print("✅ SUCCESS: Routing accuracy meets requirement (>85%)")
    else:
        print("⚠️ WARNING: Routing accuracy below requirement (>85%)")
    
    print("="*80)
    
    return accuracy >= 85


if __name__ == "__main__":
    success = test_keyword_routing()
    sys.exit(0 if success else 1)
