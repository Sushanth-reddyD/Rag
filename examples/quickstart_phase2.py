"""
Quick Start Guide for Phase 2

This is a minimal example showing how to:
1. Set up the system
2. Ingest documents
3. Perform retrieval
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def quick_start():
    """Minimal example for getting started with Phase 2."""
    
    print("="*60)
    print("Phase 2 Quick Start Guide")
    print("="*60)
    print()
    
    # Check dependencies
    try:
        import chromadb
        import sentence_transformers
        print("✅ Dependencies installed")
    except ImportError as e:
        print("❌ Missing dependencies")
        print("\nPlease install Phase 2 dependencies:")
        print("  pip install chromadb sentence-transformers")
        print("\nOr install all at once:")
        print("  pip install -r requirements-phase2.txt")
        return
    
    print()
    print("Step 1: Creating a sample document...")
    
    # Create a sample document
    docs_dir = Path("./data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    sample_doc = """
    Quick Start Guide
    
    Welcome to our product! This guide will help you get started quickly.
    
    Installation:
    Simply download the installer and follow the on-screen instructions.
    The installation process takes about 5 minutes.
    
    First Steps:
    1. Create an account
    2. Verify your email
    3. Set up your profile
    4. Start using the product
    
    Support:
    If you need help, contact our support team at support@example.com
    """
    
    sample_file = docs_dir / "quick_start.txt"
    with open(sample_file, "w") as f:
        f.write(sample_doc)
    
    print(f"✅ Created sample document: {sample_file}")
    print()
    
    # Step 2: Initialize the system
    print("Step 2: Initializing the vector database...")
    
    from src.vectordb.store import VectorStore, HybridVectorStore
    from src.ingestion.pipeline import IngestionPipeline
    
    vector_store = VectorStore(
        collection_name="quickstart",
        persist_directory="./chroma_db"
    )
    
    hybrid_store = HybridVectorStore(vector_store)
    
    pipeline = IngestionPipeline(
        vector_store=hybrid_store,
        chunking_strategy="semantic",
        chunk_size=256
    )
    
    print("✅ System initialized")
    print()
    
    # Step 3: Ingest the document
    print("Step 3: Ingesting document...")
    
    result = pipeline.ingest_document(str(sample_file))
    
    if result['status'] == 'success':
        print(f"✅ Document ingested successfully")
        print(f"   - Document ID: {result['document_id']}")
        print(f"   - Chunks created: {result['num_chunks']}")
    else:
        print(f"❌ Ingestion failed: {result.get('error')}")
        return
    
    print()
    
    # Step 4: Perform a search
    print("Step 4: Testing retrieval...")
    
    from src.retrieval.pipeline import RetrievalPipeline, RetrievalConfig
    
    config = RetrievalConfig(final_k=3)
    retrieval = RetrievalPipeline(vector_store, config)
    
    # Test queries
    queries = [
        "How do I install the product?",
        "What are the first steps?",
        "How can I get support?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        result = retrieval.retrieve(query)
        
        if result['results']:
            top_result = result['results'][0]
            print(f"   ✅ Found result (score: {top_result['score']:.3f})")
            preview = top_result['text'][:100].replace('\n', ' ')
            print(f"   📄 {preview}...")
        else:
            print("   ❌ No results found")
    
    print()
    print("="*60)
    print("✅ Quick start complete!")
    print()
    print("Next steps:")
    print("  - Add your own documents to ./data/documents/")
    print("  - Run: python examples/demo_phase2.py (full demo)")
    print("  - Run: python examples/run_orchestrator_phase2.py (with router)")
    print("  - Read: PHASE2_DOCUMENTATION.md (comprehensive guide)")
    print("="*60)


if __name__ == "__main__":
    quick_start()
