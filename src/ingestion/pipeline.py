"""Main ingestion pipeline orchestrating all ingestion stages."""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .preprocessing import DocumentPreprocessor
from .chunking import DocumentChunker
from .models import DocumentMetadata, ChunkMetadata, IngestionRecord


class IngestionPipeline:
    """
    Main ingestion pipeline implementing the architecture:
    1. Source Discovery & Harvesting
    2. Preprocessing / Cleaning / Parsing
    3. Chunking / Segmentation
    4. Embedding / Vectorization (via vector store)
    5. Metadata Enrichment & Linking
    6. Storage (via hybrid vector store)
    """
    
    def __init__(
        self,
        vector_store,
        chunking_strategy: str = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_chunk_size: int = 1024
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            vector_store: HybridVectorStore instance
            chunking_strategy: Strategy for chunking ('fixed', 'sentence', 'semantic')
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            max_chunk_size: Maximum chunk size
        """
        self.vector_store = vector_store
        self.preprocessor = DocumentPreprocessor()
        self.chunker = DocumentChunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            max_chunk_size=max_chunk_size
        )
        
        self.ingestion_records: List[IngestionRecord] = []
    
    def ingest_document(
        self,
        file_path: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single document through the full pipeline.
        
        Args:
            file_path: Path to document file
            document_metadata: Optional additional metadata
            
        Returns:
            Ingestion result with document and chunk IDs
        """
        try:
            # Stage 1: Preprocessing / Parsing
            text, parsed_metadata = self.preprocessor.process_document(file_path)
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Merge metadata
            full_metadata = {
                **parsed_metadata,
                **(document_metadata or {})
            }
            
            # Stage 2: Chunking
            chunks = self.chunker.chunk_document(cleaned_text, full_metadata)
            
            # Stage 3: Metadata Enrichment
            # Add chunk-level metadata
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                # Link chunks to each other
                if i > 0:
                    chunk.metadata['previous_chunk_id'] = f"{full_metadata['document_id']}_chunk_{i-1}"
                if i < len(chunks) - 1:
                    chunk.metadata['next_chunk_id'] = f"{full_metadata['document_id']}_chunk_{i+1}"
                
                enriched_chunks.append(chunk)
            
            # Stage 4 & 5: Embedding & Storage
            chunk_ids = self.vector_store.add_chunks(enriched_chunks, full_metadata)
            
            # Persist
            self.vector_store.vector_store.persist()
            
            return {
                'status': 'success',
                'document_id': full_metadata['document_id'],
                'num_chunks': len(chunks),
                'chunk_ids': chunk_ids,
                'metadata': full_metadata
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'file_path': file_path
            }
    
    def ingest_batch(
        self,
        file_paths: List[str],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionRecord:
        """
        Ingest a batch of documents.
        
        Args:
            file_paths: List of document file paths
            batch_metadata: Optional metadata for the batch
            
        Returns:
            IngestionRecord tracking the batch
        """
        batch_id = str(uuid.uuid4())
        
        record = IngestionRecord(
            batch_id=batch_id,
            document_ids=[],
            total_documents=len(file_paths),
            status='processing'
        )
        
        total_chunks = 0
        
        for file_path in file_paths:
            result = self.ingest_document(file_path, batch_metadata)
            
            if result['status'] == 'success':
                record.document_ids.append(result['document_id'])
                record.processed_documents += 1
                total_chunks += result['num_chunks']
            else:
                print(f"Error ingesting {file_path}: {result.get('error')}")
        
        record.total_chunks = total_chunks
        record.completed_at = datetime.now()
        record.status = 'completed' if record.processed_documents == record.total_documents else 'failed'
        
        self.ingestion_records.append(record)
        
        return record
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> IngestionRecord:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            file_patterns: Optional list of file patterns to match (e.g., ['*.pdf', '*.txt'])
            
        Returns:
            IngestionRecord tracking the batch
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Collect files
        file_paths = []
        
        if file_patterns is None:
            file_patterns = ['*.txt', '*.pdf', '*.html', '*.htm', '*.docx']
        
        for pattern in file_patterns:
            if recursive:
                file_paths.extend([str(f) for f in dir_path.rglob(pattern)])
            else:
                file_paths.extend([str(f) for f in dir_path.glob(pattern)])
        
        print(f"Found {len(file_paths)} documents in {directory}")
        
        return self.ingest_batch(file_paths, {'source_directory': str(dir_path)})
    
    def load_single_document(self, file_path: str) -> List[ChunkMetadata]:
        """
        Load a single document and return chunks (for document_loader.py)
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of ChunkMetadata objects created from the document
        """
        try:
            # Stage 1: Preprocessing / Parsing
            text, parsed_metadata = self.preprocessor.process_document(file_path)
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Stage 2: Chunking
            chunks = self.chunker.chunk_document(cleaned_text, parsed_metadata)
            
            # Stage 3: Metadata Enrichment
            enriched_chunks = []
            for i, chunk in enumerate(chunks):
                # Link chunks to each other
                if i > 0:
                    chunk.metadata['previous_chunk_id'] = f"{parsed_metadata['document_id']}_chunk_{i-1}"
                if i < len(chunks) - 1:
                    chunk.metadata['next_chunk_id'] = f"{parsed_metadata['document_id']}_chunk_{i+1}"
                
                enriched_chunks.append(chunk)
            
            # Stage 4 & 5: Embedding & Storage
            chunk_ids = self.vector_store.add_chunks(enriched_chunks, parsed_metadata)
            
            # Persist
            self.vector_store.vector_store.persist()
            
            return enriched_chunks
            
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingestion operations.
        
        Returns:
            Statistics dict
        """
        total_batches = len(self.ingestion_records)
        total_documents = sum(r.processed_documents for r in self.ingestion_records)
        total_chunks = sum(r.total_chunks for r in self.ingestion_records)
        
        successful_batches = sum(
            1 for r in self.ingestion_records
            if r.status == 'completed'
        )
        
        return {
            'total_batches': total_batches,
            'successful_batches': successful_batches,
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'vector_store_count': self.vector_store.vector_store.count()
        }
