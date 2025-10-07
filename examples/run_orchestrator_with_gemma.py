"""
Example: Run Orchestrator with Gemma Generation

This example demonstrates the complete RAG pipeline with:
- BERT Router for query classification
- Vector DB retrieval
- Gemma 3 270M for answer generation

Author: Sushanth Reddy
Date: October 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.router.orchestrator import LangGraphOrchestrator


def main():
    """Run the orchestrator with Gemma generation enabled."""
    
    print("\n" + "="*80)
    print("🚀 Customer Support RAG System with Gemma 3 270M")
    print("="*80 + "\n")
    
    print("Initializing system...")
    print("  - Loading BERT router...")
    print("  - Connecting to vector database...")
    print("  - Loading Gemma 3 270M model...")
    print()
    
    # Initialize orchestrator with all features enabled
    orchestrator = LangGraphOrchestrator(
        use_real_retrieval=True,      # Use vector DB for retrieval
        auto_load_docs=True,           # Auto-load new documents
        use_gemma_generation=True      # Use Gemma for answer generation
    )
    
    print("\n" + "="*80)
    print("System Ready! Testing sample queries...")
    print("="*80 + "\n")
    
    # Sample queries to test
    test_queries = [
        "What is your return policy?",
        "How do I choose the right running shoe?",
        "Tell me about your shoes",
        "I want to file a complaint",
        "Hello there!",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}\n")
        
        result = orchestrator.route_query(query)
        
        routing = result.get('routing_decision', 'N/A')
        print(f"Routing: {routing}")
        
        # Show full response
        print(f"\nResponse:")
        print("-" * 80)
        print(result.get('response', 'No response'))
        print("-" * 80)


if __name__ == "__main__":
    main()
