"""
Test script for document loader with manifest tracking
"""

from src.ingestion.document_loader import DocumentLoader
import json


def main():
    print("="*70)
    print("🧪 TESTING DOCUMENT LOADER WITH MANIFEST TRACKING")
    print("="*70)
    
    # Initialize loader
    loader = DocumentLoader()
    
    # Show current manifest status
    print("\n📋 Current Manifest Status:")
    print(loader.get_manifest_summary())
    
    # Check for new documents
    new_docs = loader.get_new_documents()
    print(f"\n🔍 Found {len(new_docs)} new/modified documents:")
    for doc in new_docs:
        print(f"   - {doc.name}")
    
    # Load new documents
    print("\n" + "="*70)
    print("🚀 LOADING NEW DOCUMENTS")
    print("="*70)
    
    result = loader.load_new_documents(verbose=True)
    
    # Show results
    print("\n" + "="*70)
    print("📊 INGESTION RESULTS")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    # Show updated manifest
    print("\n" + "="*70)
    print("📋 UPDATED MANIFEST")
    print("="*70)
    print(loader.get_manifest_summary())
    
    # Test: Run again (should find no new documents)
    print("\n" + "="*70)
    print("🔄 TESTING DUPLICATE PREVENTION (Running Again)")
    print("="*70)
    
    result2 = loader.load_new_documents(verbose=True)
    print(f"\nStatus: {result2['status']}")
    print(f"New files loaded: {result2['new_files']}")


if __name__ == "__main__":
    main()
