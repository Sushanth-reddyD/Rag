"""Example usage of the LangGraph orchestrator."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.router.orchestrator import LangGraphOrchestrator


def main():
    """Run example queries through the orchestrator."""
    
    print("="*80)
    print("🚀 LangGraph Orchestrator - Phase 1: Router Implementation")
    print("="*80)
    
    # Initialize orchestrator with auto-loading enabled
    print("\n📦 Initializing orchestrator...")
    orchestrator = LangGraphOrchestrator(auto_load_docs=True)
    print("✅ Orchestrator ready!\n")
    
    # Test cases from requirements
    test_queries = [
        # Retrieval queries
        ("which product do you recommend?", "retrieval"),
        ("How do I reset my password?", "retrieval"),
        ("what places do you ship outside australia?", "retrieval"),
        ("Can i return the product i bought it not from india in india?", "retrieval"),
        ("Tell me about your company's privacy policy", "retrieval"),
        ("How do I submit a refund request?", "retrieval"),
        
        # Conversational queries
        ("Hello, how are you?", "conversational"),
        ("Thanks for your help!", "conversational"),
        
        # API call queries
        ("What's the weather in London?", "api_call"),
        ("Track my order #12345", "api_call"),
        
        # Complaint queries
        ("My product arrived broken!", "complaint"),
        ("I've been waiting for 2 weeks, this is unacceptable!", "complaint"),
        
        # Edge cases
        ("Can you help me understand your return policy? Mine is defective.", "complaint"),
        ("I need the documentation for returns", "retrieval"),
    ]
    
    correct_count = 0
    total_count = len(test_queries)
    
    for query, expected_category in test_queries:
        print("\n" + "-"*80)
        result = orchestrator.route_query(query)
        
        actual_category = result["routing_decision"]
        is_correct = actual_category == expected_category
        
        if is_correct:
            correct_count += 1
            status = "✅ CORRECT"
        else:
            status = f"❌ INCORRECT (expected: {expected_category})"
        
        print(f"\nStatus: {status}")
        print(f"Expected: {expected_category} | Actual: {actual_category}")
    
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


if __name__ == "__main__":
    main()
