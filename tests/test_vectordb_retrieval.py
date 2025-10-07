"""
Test Vector Database Retrieval
Tests that documents are loaded and can be retrieved correctly
"""

from src.vectordb.store import VectorStore, HybridVectorStore
from src.ingestion.document_loader import DocumentLoader
import json


def test_vector_store():
    """Test basic vector store functionality"""
    print("="*70)
    print("🧪 TESTING VECTOR DATABASE RETRIEVAL")
    print("="*70)
    
    # Initialize vector store
    print("\n1️⃣ Initializing Vector Store...")
    vector_store_instance = VectorStore(
        collection_name="product_docs",
        persist_directory="./chroma_db"
    )
    
    vector_store = HybridVectorStore(
        vector_store=vector_store_instance,
        metadata_store_path="./data/metadata"
    )
    
    # Check document count
    count = vector_store.vector_store.count()
    print(f"   ✅ Vector store initialized")
    print(f"   📊 Total chunks in database: {count}")
    
    if count == 0:
        print("\n   ⚠️  No documents loaded. Loading documents first...")
        loader = DocumentLoader()
        loader.load_new_documents(verbose=True)
        count = vector_store.vector_store.count()
        print(f"   📊 Total chunks after loading: {count}")
    
    # Test queries
    test_queries = [
        "What is your return policy?",
        "How do I choose the right running shoe?",
        "What is GTS in running shoes?",
        "Do you ship internationally?",
        "What shoes are good for overpronation?",
        "Tell me about Brooks Ghost shoes",
        "What is plantar fasciitis?",
        "How long do running shoes last?",
    ]
    
    print("\n" + "="*70)
    print("2️⃣ TESTING RETRIEVAL QUERIES")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: '{query}'")
        print("-" * 70)
        
        try:
            # Search with metadata
            results = vector_store.search_with_metadata(query, k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} relevant chunks")
                print()
                
                for j, result in enumerate(results, 1):
                    text = result['text']
                    score = result['score']
                    metadata = result['metadata']
                    
                    # Truncate text for display
                    display_text = text[:150] + "..." if len(text) > 150 else text
                    
                    print(f"   Result {j}:")
                    print(f"   Score: {score:.4f}")
                    print(f"   Source: {metadata.get('source', 'Unknown')}")
                    print(f"   Document: {metadata.get('document_id', 'Unknown')}")
                    print(f"   Text: {display_text}")
                    print()
            else:
                print("   ⚠️  No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test with filters
    print("\n" + "="*70)
    print("3️⃣ TESTING FILTERED RETRIEVAL")
    print("="*70)
    
    print("\n📝 Query: 'return policy' (filtered by source)")
    try:
        # First, let's see what sources we have
        results = vector_store.search_with_metadata("return policy", k=5)
        if results:
            sources = set(r['metadata'].get('source', 'Unknown') for r in results)
            print(f"   Available sources: {sources}")
            
            # Try filtering by a specific source
            for source in sources:
                if source != 'Unknown':
                    print(f"\n   Filtering by source: {source}")
                    filtered_results = vector_store.vector_store.similarity_search(
                        "return policy",
                        k=2,
                        filter_metadata={'source': source}
                    )
                    print(f"   ✅ Found {len(filtered_results)} results from {source}")
                    break
    except Exception as e:
        print(f"   ⚠️  Filtered search: {str(e)}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("4️⃣ VECTOR DATABASE STATISTICS")
    print("="*70)
    
    print(f"\n   📊 Total Documents: {count}")
    
    # Check manifest
    loader = DocumentLoader()
    manifest_summary = loader.get_manifest_summary()
    print(manifest_summary)
    
    print("\n" + "="*70)
    print("✅ VECTOR DATABASE RETRIEVAL TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_vector_store()
