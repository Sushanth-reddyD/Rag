"""Quick test script for keyword-based routing (no model download required)."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.router.router_node import RouterNode


def test_keyword_routing():
    """Test keyword-based routing without LLM."""
    
    print("="*80)
    print("🧪 Testing Keyword-Based Routing (No Model Required)")
    print("="*80)
    
    # Create router instance (won't load model initially)
    print("\n📦 Creating router instance...")
    
    # We'll directly test the keyword-based routing method
    router = RouterNode.__new__(RouterNode)
    
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
    
    print("\n🔍 Testing keyword-based routing...\n")
    
    for query, expected_category in test_queries:
        decision = router._keyword_based_routing(query)
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
