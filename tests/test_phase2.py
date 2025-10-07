"""Test suite for Phase 2: Ingestion and Retrieval."""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDocumentPreprocessing:
    """Test document preprocessing and parsing."""
    
    def test_text_parser(self):
        """Test parsing plain text files."""
        from src.ingestion.preprocessing import TextParser
        
        parser = TextParser()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nWith multiple lines.")
            temp_path = f.name
        
        try:
            text, metadata = parser.parse(temp_path)
            
            assert "This is a test document" in text
            assert metadata['source_type'] == 'txt'
            assert 'checksum' in metadata
            assert metadata['file_size'] > 0
        finally:
            os.unlink(temp_path)
    
    def test_document_preprocessor(self):
        """Test main document preprocessor."""
        from src.ingestion.preprocessing import DocumentPreprocessor
        
        preprocessor = DocumentPreprocessor()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for preprocessing.")
            temp_path = f.name
        
        try:
            text, metadata = preprocessor.process_document(temp_path)
            
            assert text == "Test content for preprocessing."
            assert 'document_id' in metadata
            assert 'source' in metadata
            assert 'ingestion_timestamp' in metadata
        finally:
            os.unlink(temp_path)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        from src.ingestion.preprocessing import DocumentPreprocessor
        
        preprocessor = DocumentPreprocessor()
        
        messy_text = "  Line 1  \n\n\n  Line 2  \n  \n  Line 3  "
        cleaned = preprocessor.clean_text(messy_text)
        
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned
        # Should not have excessive newlines
        assert "\n\n\n" not in cleaned


class TestChunking:
    """Test chunking strategies."""
    
    def test_fixed_size_chunker(self):
        """Test fixed-size chunking."""
        from src.ingestion.chunking import FixedSizeChunker
        
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        
        text = "This is a test. " * 20  # Create ~300 char text
        metadata = {'document_id': 'test_doc'}
        
        chunks = chunker.chunk(text, metadata)
        
        assert len(chunks) > 1
        assert all(len(c.text) <= 70 for c in chunks)  # Allow some variance
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
    
    def test_sentence_chunker(self):
        """Test sentence-based chunking."""
        from src.ingestion.chunking import SentenceChunker
        
        chunker = SentenceChunker(target_chunk_size=100, max_chunk_size=200)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        metadata = {'document_id': 'test_doc'}
        
        chunks = chunker.chunk(text, metadata)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text.strip()
            assert 'num_sentences' in chunk.metadata
    
    def test_semantic_chunker(self):
        """Test semantic chunking."""
        from src.ingestion.chunking import SemanticChunker
        
        chunker = SemanticChunker(target_chunk_size=100, max_chunk_size=500)
        
        text = "Test content for semantic chunking."
        metadata = {
            'document_id': 'test_doc',
            'sections': [
                {'type': 'paragraph', 'start_offset': 0, 'end_offset': len(text)}
            ]
        }
        
        chunks = chunker.chunk(text, metadata)
        
        assert len(chunks) >= 1
        assert 'section_type' in chunks[0].metadata
    
    def test_document_chunker(self):
        """Test main document chunker."""
        from src.ingestion.chunking import DocumentChunker
        
        for strategy in ['fixed', 'sentence', 'semantic']:
            chunker = DocumentChunker(strategy=strategy, chunk_size=100)
            
            text = "This is a test. " * 20
            metadata = {'document_id': 'test_doc'}
            
            chunks = chunker.chunk_document(text, metadata)
            
            assert len(chunks) > 0
            assert all(isinstance(c.text, str) for c in chunks)


class TestVectorStore:
    """Test vector store functionality."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for vector DB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_vector_store_initialization(self, temp_db_dir):
        """Test vector store can be initialized."""
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")
        
        from src.vectordb.store import VectorStore
        
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=temp_db_dir
        )
        
        assert store.collection_name == "test_collection"
        assert store.persist_directory == temp_db_dir
    
    def test_add_and_search_documents(self, temp_db_dir):
        """Test adding and searching documents."""
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")
        
        from src.vectordb.store import VectorStore
        
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=temp_db_dir
        )
        
        # Add documents
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A journey of a thousand miles begins with a single step",
            "To be or not to be, that is the question"
        ]
        metadatas = [
            {'source': 'test1', 'category': 'animals'},
            {'source': 'test2', 'category': 'philosophy'},
            {'source': 'test3', 'category': 'literature'}
        ]
        
        ids = store.add_documents(texts, metadatas)
        
        assert len(ids) == 3
        assert store.count() == 3
        
        # Search
        results = store.similarity_search("fox and dog", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


class TestRetrievalPipeline:
    """Test retrieval pipeline."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        class MockVectorStore:
            def similarity_search(self, query, k, filter_metadata=None):
                return [
                    ("Sample text 1", 0.9, {'chunk_id': 'c1', 'document_id': 'd1'}),
                    ("Sample text 2", 0.7, {'chunk_id': 'c2', 'document_id': 'd1'}),
                ]
        
        return MockVectorStore()
    
    def test_query_normalization(self):
        """Test query normalization."""
        from src.retrieval.pipeline import QueryProcessor
        
        processor = QueryProcessor()
        
        query = "  What's  THE  Return  Policy?  "
        normalized = processor.normalize_query(query)
        
        assert normalized == "whats the return policy"
    
    def test_query_expansion(self):
        """Test query expansion."""
        from src.retrieval.pipeline import QueryProcessor
        
        processor = QueryProcessor()
        
        query = "return policy"
        expanded = processor.expand_query(query)
        
        assert len(expanded) > 1
        assert query in expanded
    
    def test_retrieval_pipeline(self, mock_vector_store):
        """Test full retrieval pipeline."""
        from src.retrieval.pipeline import RetrievalPipeline, RetrievalConfig
        
        config = RetrievalConfig(
            initial_k=10,
            final_k=5,
            use_dense=True,
            use_sparse=False,
            use_reranking=True
        )
        
        pipeline = RetrievalPipeline(mock_vector_store, config)
        
        result = pipeline.retrieve("test query")
        
        assert 'results' in result
        assert 'num_results' in result
        assert 'retrieval_latency_ms' in result
        assert isinstance(result['results'], list)


class TestIngestionPipeline:
    """Test ingestion pipeline."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        db_dir = tempfile.mkdtemp()
        docs_dir = tempfile.mkdtemp()
        meta_dir = tempfile.mkdtemp()
        
        yield db_dir, docs_dir, meta_dir
        
        shutil.rmtree(db_dir)
        shutil.rmtree(docs_dir)
        shutil.rmtree(meta_dir)
    
    def test_ingestion_pipeline_init(self, temp_dirs):
        """Test ingestion pipeline initialization."""
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")
        
        db_dir, docs_dir, meta_dir = temp_dirs
        
        from src.vectordb.store import VectorStore, HybridVectorStore
        from src.ingestion.pipeline import IngestionPipeline
        
        vector_store = VectorStore(
            collection_name="test",
            persist_directory=db_dir
        )
        hybrid_store = HybridVectorStore(vector_store, meta_dir)
        
        pipeline = IngestionPipeline(
            vector_store=hybrid_store,
            chunking_strategy="fixed",
            chunk_size=100
        )
        
        assert pipeline.chunker is not None
        assert pipeline.preprocessor is not None


class TestModels:
    """Test Pydantic models."""
    
    def test_document_metadata(self):
        """Test DocumentMetadata model."""
        from src.ingestion.models import DocumentMetadata
        from datetime import datetime
        
        metadata = DocumentMetadata(
            document_id="doc123",
            source="/path/to/doc.txt",
            title="Test Document",
            source_type="txt"
        )
        
        assert metadata.document_id == "doc123"
        assert metadata.source == "/path/to/doc.txt"
        assert isinstance(metadata.ingestion_timestamp, datetime)
    
    def test_chunk_metadata(self):
        """Test ChunkMetadata model."""
        from src.ingestion.models import ChunkMetadata
        
        chunk_meta = ChunkMetadata(
            chunk_id="chunk123",
            document_id="doc123",
            chunk_text="Sample chunk text",
            chunk_index=0,
            start_offset=0,
            end_offset=100
        )
        
        assert chunk_meta.chunk_id == "chunk123"
        assert chunk_meta.document_id == "doc123"
        assert chunk_meta.chunk_index == 0
    
    def test_retrieval_result(self):
        """Test RetrievalResult model."""
        from src.ingestion.models import RetrievalResult
        
        result = RetrievalResult(
            chunk_id="chunk123",
            document_id="doc123",
            chunk_text="Sample text",
            score=0.95,
            rank=1,
            retrieval_method="dense",
            confidence="high"
        )
        
        assert result.score == 0.95
        assert result.confidence == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
