"""Data models for document ingestion and metadata."""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    
    document_id: str = Field(description="Unique document identifier")
    source: str = Field(description="URL, file path, or repository location")
    title: Optional[str] = Field(default=None, description="Document title")
    authors: Optional[List[str]] = Field(default=None, description="Document authors")
    publication_date: Optional[datetime] = Field(default=None, description="Publication date")
    version: Optional[str] = Field(default="1.0", description="Document version")
    domain: Optional[str] = Field(default=None, description="Domain/category")
    source_type: Literal["pdf", "html", "docx", "txt", "api", "web"] = Field(
        default="txt",
        description="Type of source document"
    )
    citation_ids: Optional[List[str]] = Field(
        default=None,
        description="IDs of documents cited by this document"
    )
    license: Optional[str] = Field(default=None, description="License/access rights")
    abstract: Optional[str] = Field(default=None, description="Document abstract/summary")
    checksum: Optional[str] = Field(default=None, description="Document checksum for change detection")
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When document was ingested"
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )
    custom_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional custom metadata"
    )


class ChunkMetadata(BaseModel):
    """Chunk-level metadata."""
    
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document ID")
    chunk_text: str = Field(description="The actual chunk text content")
    chunk_index: int = Field(description="Order of chunk within document")
    start_offset: int = Field(description="Starting character offset in document")
    end_offset: int = Field(description="Ending character offset in document")
    
    # Structural information
    section_header: Optional[str] = Field(
        default=None,
        description="Section or heading this chunk belongs to"
    )
    structural_tags: Optional[List[str]] = Field(
        default=None,
        description="Tags like 'paragraph', 'table', 'code', 'equation'"
    )
    
    # Embedding information
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model used"
    )
    embedding_model_version: str = Field(
        default="1.0",
        description="Version of embedding model"
    )
    
    # Context and relationships
    overlap_with_previous: int = Field(
        default=0,
        description="Character overlap with previous chunk"
    )
    overlap_with_next: int = Field(
        default=0,
        description="Character overlap with next chunk"
    )
    previous_chunk_id: Optional[str] = Field(
        default=None,
        description="Reference to previous chunk"
    )
    next_chunk_id: Optional[str] = Field(
        default=None,
        description="Reference to next chunk"
    )
    
    # Citation and importance
    citation_pointers: Optional[List[str]] = Field(
        default=None,
        description="Citations within this chunk"
    )
    importance_score: Optional[float] = Field(
        default=None,
        description="Local relevance/importance score"
    )
    
    # Timestamps
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When chunk was created"
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )


class IngestionRecord(BaseModel):
    """Record for tracking ingestion operations."""
    
    batch_id: str = Field(description="Unique batch identifier")
    document_ids: List[str] = Field(description="Documents in this batch")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        default="pending",
        description="Ingestion status"
    )
    total_documents: int = Field(description="Total documents in batch")
    processed_documents: int = Field(default=0, description="Documents processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="Batch start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Batch completion time"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if failed"
    )


class RetrievalResult(BaseModel):
    """Result from retrieval pipeline."""
    
    chunk_id: str = Field(description="Retrieved chunk ID")
    document_id: str = Field(description="Source document ID")
    chunk_text: str = Field(description="Chunk text content")
    score: float = Field(description="Relevance score")
    
    # Metadata for citation
    document_metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Full document metadata"
    )
    chunk_metadata: Optional[ChunkMetadata] = Field(
        default=None,
        description="Full chunk metadata"
    )
    
    # Ranking information
    rank: int = Field(description="Rank in result list")
    retrieval_method: Literal["dense", "sparse", "hybrid", "reranked"] = Field(
        description="Method used for retrieval"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence in result relevance"
    )


class QueryMetadata(BaseModel):
    """Metadata for tracking queries."""
    
    query_id: str = Field(description="Unique query identifier")
    query_text: str = Field(description="Original query text")
    normalized_query: Optional[str] = Field(
        default=None,
        description="Normalized/expanded query"
    )
    query_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Query timestamp"
    )
    
    # Results
    num_results: int = Field(default=0, description="Number of results returned")
    retrieval_latency_ms: Optional[float] = Field(
        default=None,
        description="Retrieval latency in milliseconds"
    )
    
    # Filters applied
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters applied"
    )
