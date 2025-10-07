"""
Document Loader with Manifest Tracking
Loads documents into vector database while preventing duplicates
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

from src.ingestion.pipeline import IngestionPipeline
from src.vectordb.store import HybridVectorStore


class DocumentLoader:
    """
    Manages document ingestion with duplicate prevention via manifest tracking
    """
    
    def __init__(
        self,
        documents_dir: str = "./data/documents",
        manifest_path: str = "./data/metadata/ingestion_manifest.json",
        vectordb_path: str = "./chroma_db"
    ):
        """
        Initialize the document loader
        
        Args:
            documents_dir: Directory containing documents to load
            manifest_path: Path to JSON file tracking loaded documents
            vectordb_path: Path to ChromaDB storage
        """
        self.documents_dir = Path(documents_dir)
        self.manifest_path = Path(manifest_path)
        self.vectordb_path = vectordb_path
        
        # Ensure manifest directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing manifest
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load the ingestion manifest from JSON file"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "1.0",
                "last_updated": None,
                "loaded_files": {},
                "statistics": {
                    "total_files_loaded": 0,
                    "total_chunks_created": 0,
                    "last_ingestion_date": None
                }
            }
    
    def _save_manifest(self):
        """Save the ingestion manifest to JSON file"""
        self.manifest["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_document_files(self) -> List[Path]:
        """
        Get all document files from documents directory
        
        Returns:
            List of document file paths
        """
        if not self.documents_dir.exists():
            print(f"⚠️  Documents directory not found: {self.documents_dir}")
            return []
        
        # Supported file extensions
        extensions = ['.txt', '.md', '.pdf', '.docx']
        
        files = []
        for ext in extensions:
            files.extend(self.documents_dir.glob(f'*{ext}'))
        
        return sorted(files)
    
    def _is_file_loaded(self, file_path: Path) -> bool:
        """
        Check if file has been loaded and hasn't changed
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is already loaded and unchanged
        """
        file_name = file_path.name
        
        if file_name not in self.manifest["loaded_files"]:
            return False
        
        # Check if file hash has changed (file was modified)
        current_hash = self._compute_file_hash(file_path)
        stored_hash = self.manifest["loaded_files"][file_name].get("file_hash")
        
        if current_hash != stored_hash:
            print(f"📝 File modified, will reload: {file_name}")
            return False
        
        return True
    
    def get_new_documents(self) -> List[Path]:
        """
        Get list of documents that need to be loaded
        
        Returns:
            List of new or modified document paths
        """
        all_files = self._get_document_files()
        new_files = [f for f in all_files if not self._is_file_loaded(f)]
        return new_files
    
    def load_new_documents(self, verbose: bool = True) -> Dict:
        """
        Load only new or modified documents into vector database
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with ingestion statistics
        """
        new_files = self.get_new_documents()
        
        if not new_files:
            if verbose:
                print("✅ All documents already loaded. No new files to ingest.")
            return {
                "new_files": 0,
                "files_loaded": [],
                "chunks_created": 0,
                "status": "up_to_date"
            }
        
        if verbose:
            print(f"\n📚 Found {len(new_files)} new/modified document(s) to load:")
            for f in new_files:
                print(f"   - {f.name}")
            print()
        
        # Initialize vector store
        from src.vectordb.store import VectorStore
        vector_store_instance = VectorStore(
            collection_name="product_docs",
            persist_directory=self.vectordb_path
        )
        
        vector_store = HybridVectorStore(
            vector_store=vector_store_instance,
            metadata_store_path="./data/metadata"
        )
        
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(
            vector_store=vector_store,
            chunking_strategy="semantic",
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Track statistics
        total_chunks = 0
        loaded_files = []
        
        # Process each new file
        for file_path in new_files:
            try:
                if verbose:
                    print(f"🔄 Loading: {file_path.name}...")
                
                # Load and process single file
                chunks = pipeline.load_single_document(str(file_path))
                
                if chunks:
                    # Store file metadata in manifest
                    file_hash = self._compute_file_hash(file_path)
                    file_stats = os.stat(file_path)
                    
                    self.manifest["loaded_files"][file_path.name] = {
                        "file_path": str(file_path),
                        "file_hash": file_hash,
                        "file_size_bytes": file_stats.st_size,
                        "chunks_created": len(chunks),
                        "loaded_at": datetime.now().isoformat(),
                        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }
                    
                    total_chunks += len(chunks)
                    loaded_files.append(file_path.name)
                    
                    if verbose:
                        print(f"   ✅ Created {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error loading {file_path.name}: {str(e)}")
                continue
        
        # Update global statistics
        self.manifest["statistics"]["total_files_loaded"] = len(self.manifest["loaded_files"])
        self.manifest["statistics"]["total_chunks_created"] += total_chunks
        self.manifest["statistics"]["last_ingestion_date"] = datetime.now().isoformat()
        
        # Save updated manifest
        self._save_manifest()
        
        if verbose:
            print(f"\n✅ Ingestion complete!")
            print(f"   📊 New files loaded: {len(loaded_files)}")
            print(f"   📊 New chunks created: {total_chunks}")
            print(f"   📊 Total files in database: {self.manifest['statistics']['total_files_loaded']}")
            print(f"   📁 Manifest saved: {self.manifest_path}\n")
        
        return {
            "new_files": len(loaded_files),
            "files_loaded": loaded_files,
            "chunks_created": total_chunks,
            "status": "success"
        }
    
    def get_manifest_summary(self) -> str:
        """
        Get human-readable summary of manifest
        
        Returns:
            Formatted string with manifest information
        """
        stats = self.manifest["statistics"]
        
        summary = f"""
📊 Document Ingestion Manifest Summary
{'='*50}
Total Files Loaded: {stats['total_files_loaded']}
Total Chunks Created: {stats['total_chunks_created']}
Last Ingestion: {stats['last_ingestion_date'] or 'Never'}
Manifest Path: {self.manifest_path}

Loaded Files:
"""
        
        for file_name, info in self.manifest["loaded_files"].items():
            summary += f"  ✓ {file_name} ({info['chunks_created']} chunks, loaded {info['loaded_at'][:10]})\n"
        
        return summary
    
    def remove_file_from_manifest(self, file_name: str) -> bool:
        """
        Remove a file from manifest (forces reload on next run)
        
        Args:
            file_name: Name of file to remove from manifest
            
        Returns:
            True if file was removed, False if not found
        """
        if file_name in self.manifest["loaded_files"]:
            del self.manifest["loaded_files"][file_name]
            self.manifest["statistics"]["total_files_loaded"] = len(self.manifest["loaded_files"])
            self._save_manifest()
            print(f"✅ Removed {file_name} from manifest. Will be reloaded on next ingestion.")
            return True
        else:
            print(f"⚠️  {file_name} not found in manifest.")
            return False
    
    def clear_manifest(self) -> bool:
        """
        Clear all entries from manifest (forces full reload)
        
        Returns:
            True if successful
        """
        self.manifest = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "loaded_files": {},
            "statistics": {
                "total_files_loaded": 0,
                "total_chunks_created": 0,
                "last_ingestion_date": None
            }
        }
        self._save_manifest()
        print("✅ Manifest cleared. All documents will be reloaded on next ingestion.")
        return True


def auto_load_documents(verbose: bool = True) -> Dict:
    """
    Convenience function to automatically load new documents
    Called by orchestrator on startup
    
    Args:
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary with ingestion statistics
    """
    loader = DocumentLoader()
    return loader.load_new_documents(verbose=verbose)


if __name__ == "__main__":
    # Test the document loader
    print("🚀 Testing Document Loader with Manifest Tracking\n")
    
    loader = DocumentLoader()
    
    # Show current manifest
    print(loader.get_manifest_summary())
    
    # Load new documents
    result = loader.load_new_documents(verbose=True)
    
    print("\n" + "="*50)
    print("📋 Ingestion Result:")
    print(json.dumps(result, indent=2))
