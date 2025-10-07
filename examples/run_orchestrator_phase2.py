"""
Example: Orchestrator with Phase 2 Retrieval Integration

This demonstrates the full system with router + real retrieval agent.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    """Run orchestrator with Phase 2 retrieval."""
    print("="*80)
    print("LangGraph Orchestrator with Phase 2 Retrieval")
    print("="*80)
    print()
    
    # Check if Phase 2 demo has been run
    from pathlib import Path
    if not Path("./chroma_db").exists():
        print("⚠️  Vector database not found!")
        print()
        print("Please run the Phase 2 demo first to ingest documents:")
        print("   python examples/demo_phase2.py")
        print()
        return
    
    # Initialize orchestrator with real retrieval
    print("Initializing orchestrator with real retrieval agent...")
    try:
        from src.router.orchestrator import LangGraphOrchestrator
        
        # Create orchestrator with real retrieval enabled
        orchestrator = LangGraphOrchestrator(use_real_retrieval=True)
        print("✅ Orchestrator initialized")
        print()
        
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}")
        print()
        print("Make sure Phase 2 dependencies are installed:")
        print("   pip install -r requirements-phase2.txt")
        return
    
    # Test queries across all categories
    test_cases = [
        # Retrieval queries (will use vector DB)
        {
            "query": "What is your return policy?",
            "expected": "retrieval",
            "description": "Policy question"
        },
        {
            "query": "How do I reset my password?",
            "expected": "retrieval",
            "description": "FAQ question"
        },
        {
            "query": "Tell me about warranty coverage",
            "expected": "retrieval",
            "description": "Documentation query"
        },
        # Other categories (placeholders)
        {
            "query": "Hello, how are you?",
            "expected": "conversational",
            "description": "Greeting"
        },
        {
            "query": "Track my order #12345",
            "expected": "api_call",
            "description": "Order tracking"
        },
        {
            "query": "My product arrived broken!",
            "expected": "complaint",
            "description": "Customer complaint"
        },
    ]
    
    print("Testing orchestrator with various queries...")
    print()
    
    correct = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{total}: {test['description']}")
        print(f"Query: \"{test['query']}\"")
        print(f"Expected category: {test['expected']}")
        print("-" * 80)
        
        # Route the query
        result = orchestrator.route_query(test['query'])
        
        # Check result
        actual = result.get('routing_decision', '')
        is_correct = actual == test['expected']
        
        if is_correct:
            correct += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"\n{status}")
        print(f"Actual category: {actual}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        
        # Display response (especially for retrieval)
        if actual == "retrieval" and 'retrieval_results' in result:
            print(f"\n📄 Retrieved {len(result['retrieval_results'])} results:")
            metadata = result.get('retrieval_metadata', {})
            print(f"   Latency: {metadata.get('latency_ms', 0):.2f}ms")
            print(f"   Total documents in DB: {metadata.get('total_documents', 0)}")
            
            for res in result['retrieval_results'][:2]:  # Show top 2
                print(f"\n   Result {res['rank']}: Score {res['score']:.3f}")
                text_preview = res['text'][:100].replace('\n', ' ')
                print(f"   {text_preview}...")
        
        print(f"\nResponse preview:")
        response = result.get('response', '')[:300]
        print(f"{response}...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {total}")
    print(f"Passed: {correct}")
    print(f"Failed: {total - correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print()
    
    # Show retrieval stats
    if orchestrator.use_real_retrieval and orchestrator.retrieval_agent:
        stats = orchestrator.retrieval_agent.get_stats()
        print("Retrieval System Stats:")
        print(f"  - Documents in vector DB: {stats['total_documents']}")
        print(f"  - Collection: {stats['collection_name']}")
        print(f"  - Embedding model: {stats['embedding_model']}")
    
    print("="*80)


if __name__ == "__main__":
    main()
