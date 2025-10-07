"""
Test script for BERT Routing + Retrieval + Gemma Generation Pipeline

This script tests the complete RAG pipeline:
1. BERT fine-tuned router classifies the query
2. If routed to retrieval, vector DB retrieves relevant chunks
3. Gemma 3 270M generates natural language answer from context

Author: Sushanth Reddy
Date: October 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.router.orchestrator import LangGraphOrchestrator


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def test_gemma_generation():
    """Test the complete BERT + Retrieval + Gemma pipeline."""
    
    print("\n" + "="*80)
    print("🧪 TESTING: BERT Router + Retrieval + Gemma 3 270M Generation")
    print("="*80 + "\n")
    
    # Test queries
    test_queries = [
        # Retrieval queries (should route to retrieval and generate answers)
        "What is your return policy?",
        "How do I choose the right Brooks shoe?",
        "What is GTS in running shoes?",
        "Tell me about Brooks Ghost shoes",
        "Do you have shoes for overpronation?",
        "What is plantar fasciitis?",
        
        # Non-retrieval queries (should route elsewhere)
        "I want to file a complaint",
        "Hello, how are you?",
    ]
    
    print("🔧 Initializing orchestrator with:")
    print("  - BERT Router: Fine-tuned for query classification")
    print("  - Vector Retrieval: ChromaDB with 82 chunks")
    print("  - Gemma 3 270M: Answer generation\n")
    
    # Initialize orchestrator with Gemma generation
    try:
        orchestrator = LangGraphOrchestrator(
            use_real_retrieval=True,
            auto_load_docs=True,
            use_gemma_generation=True  # Enable Gemma!
        )
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}")
        print("\nMake sure to install dependencies:")
        print("  pip install transformers accelerate torch")
        return
    
    print("\n" + "="*80)
    print("Running Test Queries")
    print("="*80 + "\n")
    
    # Test each query
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"{'─'*80}")
        
        try:
            result = orchestrator.route_query(query)
            
            routing = result.get('routing_decision', 'N/A')
            confidence = result.get('confidence', 'N/A')
            response = result.get('response', 'No response')
            
            print(f"\n✓ Routing: {routing}")
            print(f"✓ Confidence: {confidence}")
            
            # If retrieval route, show generated answer
            if routing == 'retrieval' and 'generated_answer' in result:
                print(f"\n📝 GENERATED ANSWER:")
                print("-" * 80)
                print(result['generated_answer'])
                print("-" * 80)
                
                # Show retrieval metadata
                metadata = result.get('retrieval_metadata', {})
                print(f"\n📊 Retrieval Stats:")
                print(f"  - Results: {metadata.get('num_results', 0)}")
                print(f"  - Latency: {metadata.get('latency_ms', 0):.2f}ms")
                print(f"  - Total Docs: {metadata.get('total_documents', 0)}")
            else:
                print(f"\n💬 Response:")
                print("-" * 80)
                print(response[:500] + "..." if len(response) > 500 else response)
                print("-" * 80)
            
            results.append({
                'query': query,
                'routing': routing,
                'success': True
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                'query': query,
                'routing': 'error',
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80 + "\n")
    
    successful = sum(1 for r in results if r['success'])
    retrieval_queries = sum(1 for r in results if r['routing'] == 'retrieval')
    
    print(f"Total Queries: {len(test_queries)}")
    print(f"Successful: {successful}/{len(test_queries)} ({successful/len(test_queries)*100:.1f}%)")
    print(f"Routed to Retrieval: {retrieval_queries}")
    print(f"Generated Answers: {retrieval_queries} (with Gemma 3 270M)")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80 + "\n")
    
    print("Pipeline Architecture:")
    print("  1. User Query → BERT Router (Fine-tuned)")
    print("  2. If 'retrieval' → Vector DB (ChromaDB + MiniLM)")
    print("  3. Retrieved Context → Gemma 3 270M")
    print("  4. Natural Language Answer → User")
    print()


if __name__ == "__main__":
    test_gemma_generation()
