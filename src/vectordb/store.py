"""Vector database integration using ChromaDB (CPU-optimized)."""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

from pydantic import BaseModel


class VectorStore:
    """
    CPU-optimized vector store using ChromaDB.
    Handles vector storage, indexing, and similarity search.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Lazy imports for optional dependencies
        self._client = None
        self._collection = None
        self._embedding_function = None
    
    def _initialize_client(self):
        """Initialize ChromaDB client (lazy initialization)."""
        if self._client is not None:
            return
        
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize client with new API (PersistentClient)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model_name}
        )
    
    def _initialize_embeddings(self):
        """Initialize embedding model (lazy initialization)."""
        if self._embedding_function is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        # Use CPU-optimized model
        self._embedding_function = SentenceTransformer(
            self.embedding_model_name,
            device='cpu'
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dicts for each chunk
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        self._initialize_client()
        self._initialize_embeddings()
        
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadata is JSON-serializable
        clean_metadatas = []
        for metadata in metadatas:
            clean_meta = {}
            for key, value in metadata.items():
                # Convert datetime to string
                if hasattr(value, 'isoformat'):
                    clean_meta[key] = value.isoformat()
                # Convert lists/dicts to JSON strings if needed
                elif isinstance(value, (list, dict)):
                    clean_meta[key] = json.dumps(value)
                # Keep simple types as-is
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            clean_metadatas.append(clean_meta)
        
        # Compute embeddings
        embeddings = self._embedding_function.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection
        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metadatas,
            ids=ids
        )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples (text, score, metadata)
        """
        self._initialize_client()
        self._initialize_embeddings()
        
        # Compute query embedding
        query_embedding = self._embedding_function.encode(
            [query],
            convert_to_numpy=True
        ).tolist()[0]
        
        # Search with optional filtering
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Convert distance to similarity score (0-1)
            # ChromaDB uses L2 distance, lower is better
            similarity_score = 1 / (1 + distance)
            
            formatted_results.append((text, similarity_score, metadata))
        
        return formatted_results
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self._initialize_client()
        
        if ids:
            self._collection.delete(ids=ids)
    
    def update_document(
        self,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a document.
        
        Args:
            id: Document ID
            text: New text (if updating)
            metadata: New metadata (if updating)
        """
        self._initialize_client()
        self._initialize_embeddings()
        
        # If updating text, recompute embedding
        embedding = None
        if text is not None:
            embedding = self._embedding_function.encode(
                [text],
                convert_to_numpy=True
            ).tolist()[0]
        
        # Update document
        update_params = {'ids': [id]}
        if text is not None:
            update_params['documents'] = [text]
            update_params['embeddings'] = [embedding]
        if metadata is not None:
            # Clean metadata
            clean_meta = {}
            for key, value in metadata.items():
                if hasattr(value, 'isoformat'):
                    clean_meta[key] = value.isoformat()
                elif isinstance(value, (list, dict)):
                    clean_meta[key] = json.dumps(value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            update_params['metadatas'] = [clean_meta]
        
        self._collection.update(**update_params)
    
    def get_document(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            id: Document ID
            
        Returns:
            Document dict or None if not found
        """
        self._initialize_client()
        
        results = self._collection.get(ids=[id])
        
        if results['documents']:
            return {
                'id': id,
                'text': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        
        return None
    
    def count(self) -> int:
        """Get number of documents in collection."""
        self._initialize_client()
        return self._collection.count()
    
    def persist(self) -> None:
        """Persist the database to disk (automatic with PersistentClient)."""
        # PersistentClient automatically persists changes
        # This method is kept for backward compatibility
        pass
    
    def reset(self) -> None:
        """Delete all documents in the collection."""
        self._initialize_client()
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model_name}
        )


class HybridVectorStore:
    """
    Hybrid storage combining vector store and metadata store.
    Optimized for both similarity search and metadata filtering.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store_path: str = "./data/metadata"
    ):
        """
        Initialize hybrid store.
        
        Args:
            vector_store: Vector store instance
            metadata_store_path: Path to store detailed metadata
        """
        self.vector_store = vector_store
        self.metadata_store_path = Path(metadata_store_path)
        self.metadata_store_path.mkdir(parents=True, exist_ok=True)
    
    def add_chunks(
        self,
        chunks: List[Any],  # List of Chunk objects from chunking.py
        document_metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Add chunks with full metadata tracking.
        
        Args:
            chunks: List of Chunk objects
            document_metadata: Document-level metadata
            
        Returns:
            List of chunk IDs
        """
        texts = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk_id = f"{document_metadata['document_id']}_chunk_{chunk.chunk_index}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            
            # Store minimal metadata in vector store
            vector_metadata = {
                'chunk_id': chunk_id,
                'document_id': document_metadata['document_id'],
                'chunk_index': chunk.chunk_index,
                'source': document_metadata.get('source', ''),
                'title': document_metadata.get('title', ''),
                'domain': document_metadata.get('domain', ''),
            }
            metadatas.append(vector_metadata)
            
            # Store full metadata separately
            full_metadata = {
                **vector_metadata,
                **chunk.metadata,
                'text': chunk.text,
                'start_offset': chunk.start_offset,
                'end_offset': chunk.end_offset,
                'document_metadata': document_metadata
            }
            
            # Save to metadata store
            metadata_file = self.metadata_store_path / f"{chunk_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
        
        # Add to vector store
        return self.vector_store.add_documents(texts, metadatas, ids)
    
    def search_with_metadata(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with full metadata retrieval.
        
        Args:
            query: Query text
            k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dicts with full metadata
        """
        # Search vector store
        results = self.vector_store.similarity_search(
            query, k, filter_metadata
        )
        
        # Enhance with full metadata
        enhanced_results = []
        for text, score, minimal_metadata in results:
            chunk_id = minimal_metadata.get('chunk_id')
            
            # Load full metadata
            metadata_file = self.metadata_store_path / f"{chunk_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    full_metadata = json.load(f)
            else:
                full_metadata = minimal_metadata
            
            enhanced_results.append({
                'text': text,
                'score': score,
                'metadata': full_metadata
            })
        
        return enhanced_results
