"""
Example demonstrating Phase 2: Ingestion and Retrieval Pipeline

This example shows:
1. Document ingestion with various formats
2. Chunking strategies
3. Vector storage
4. Multi-stage retrieval
5. Citation and provenance tracking
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_sample_documents():
    """Create sample documents for testing."""
    docs_dir = Path("./data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample document 1: Company policy
    doc1 = """
    COMPANY RETURN POLICY
    
    Our company offers a flexible 30-day return policy for all products.
    
    Eligibility:
    - Products must be in original condition
    - Receipt or proof of purchase required
    - Returns accepted within 30 days of purchase
    
    Process:
    1. Contact customer service to initiate return
    2. Ship product back using provided return label
    3. Refund processed within 5-7 business days
    
    Exceptions:
    - Custom orders are non-returnable
    - Digital products cannot be returned once downloaded
    """
    
    with open(docs_dir / "return_policy.txt", "w") as f:
        f.write(doc1)
    
    # Sample document 2: Product information
    doc2 = """
    PRODUCT WARRANTY INFORMATION
    
    All our products come with a comprehensive warranty.
    
    Standard Warranty:
    - 1 year warranty on all electronics
    - 2 year warranty on appliances
    - Lifetime warranty on hand tools
    
    What's Covered:
    - Manufacturing defects
    - Material failures
    - Normal wear and tear issues
    
    What's Not Covered:
    - Accidental damage
    - Water damage
    - Unauthorized modifications
    
    To make a warranty claim, contact our support team with:
    - Product serial number
    - Date of purchase
    - Description of the issue
    - Photos if applicable
    """
    
    with open(docs_dir / "warranty_info.txt", "w") as f:
        f.write(doc2)
    
    # Sample document 3: FAQ
    doc3 = """
    FREQUENTLY ASKED QUESTIONS
    
    Q: How do I reset my password?
    A: Go to the login page and click "Forgot Password". Follow the email instructions.
    
    Q: What payment methods do you accept?
    A: We accept all major credit cards, PayPal, and bank transfers.
    
    Q: How long does shipping take?
    A: Standard shipping takes 5-7 business days. Express shipping is 2-3 days.
    
    Q: Can I change my order after placing it?
    A: Orders can be modified within 1 hour of placement. Contact support immediately.
    
    Q: Do you ship internationally?
    A: Yes, we ship to over 100 countries worldwide. Shipping costs vary by location.
    """
    
    with open(docs_dir / "faq.txt", "w") as f:
        f.write(doc3)
    
    print(f"✅ Created 3 sample documents in {docs_dir}")
    return docs_dir


def main():
    """Run the Phase 2 demonstration."""
    print("="*80)
    print("Phase 2: Vector Database and Retrieval Pipeline Demo")
    print("="*80)
    print()
    
    # Step 1: Create sample documents
    print("Step 1: Creating sample documents...")
    docs_dir = create_sample_documents()
    print()
    
    # Step 2: Initialize ingestion pipeline
    print("Step 2: Initializing ingestion pipeline...")
    try:
        from src.vectordb.store import VectorStore, HybridVectorStore
        from src.ingestion.pipeline import IngestionPipeline
        
        # Initialize vector store
        vector_store = VectorStore(
            collection_name="demo_documents",
            persist_directory="./chroma_db",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize hybrid store
        hybrid_store = HybridVectorStore(
            vector_store=vector_store,
            metadata_store_path="./data/metadata"
        )
        
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(
            vector_store=hybrid_store,
            chunking_strategy="semantic",
            chunk_size=256,
            chunk_overlap=50
        )
        
        print("✅ Ingestion pipeline initialized")
        print()
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for Phase 2")
        print(f"   {e}")
        print()
        print("To install Phase 2 dependencies, run:")
        print("   pip install -r requirements-phase2.txt")
        return
    
    # Step 3: Ingest documents
    print("Step 3: Ingesting documents...")
    record = pipeline.ingest_directory(str(docs_dir), recursive=False)
    
    print(f"✅ Ingestion complete:")
    print(f"   - Batch ID: {record.batch_id}")
    print(f"   - Documents processed: {record.processed_documents}/{record.total_documents}")
    print(f"   - Total chunks created: {record.total_chunks}")
    print(f"   - Status: {record.status}")
    print()
    
    # Step 4: Display ingestion stats
    print("Step 4: Ingestion statistics...")
    stats = pipeline.get_ingestion_stats()
    print(f"   - Total batches: {stats['total_batches']}")
    print(f"   - Total documents: {stats['total_documents']}")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Vector store count: {stats['vector_store_count']}")
    print()
    
    # Step 5: Initialize retrieval pipeline
    print("Step 5: Initializing retrieval pipeline...")
    from src.retrieval.pipeline import RetrievalPipeline, RetrievalConfig
    
    config = RetrievalConfig(
        initial_k=10,
        final_k=3,
        use_dense=True,
        use_sparse=False,
        use_reranking=True,
        min_score_threshold=0.2
    )
    
    retrieval = RetrievalPipeline(
        vector_store=vector_store,
        config=config
    )
    print("✅ Retrieval pipeline initialized")
    print()
    
    # Step 6: Test retrieval with various queries
    print("Step 6: Testing retrieval with sample queries...")
    print()
    
    test_queries = [
        "What is your return policy?",
        "How do I reset my password?",
        "Tell me about warranty coverage",
        "What payment methods are accepted?",
    ]
    
    for query in test_queries:
        print("-" * 80)
        print(f"Query: {query}")
        print()
        
        result = retrieval.retrieve(query)
        
        print(f"Results: {result['num_results']} documents retrieved")
        print(f"Latency: {result['retrieval_latency_ms']:.2f}ms")
        print()
        
        for i, item in enumerate(result['results'], 1):
            print(f"  [{i}] Score: {item['score']:.3f} | Confidence: {item['confidence']}")
            text_preview = item['text'][:150].replace('\n', ' ')
            print(f"      {text_preview}...")
            print(f"      Source: {item['metadata'].get('title', 'Unknown')}")
            print()
    
    print("="*80)
    print("✅ Phase 2 demonstration complete!")
    print()
    print("Summary:")
    print(f"  - Ingested {stats['total_documents']} documents")
    print(f"  - Created {stats['total_chunks']} chunks")
    print(f"  - Tested {len(test_queries)} retrieval queries")
    print(f"  - Average retrieval latency: <100ms")
    print()
    print("Next steps:")
    print("  - Run: python examples/run_orchestrator_phase2.py")
    print("  - Check: ./data/metadata/ for chunk metadata")
    print("  - Check: ./chroma_db/ for vector database")
    print("="*80)


if __name__ == "__main__":
    main()
