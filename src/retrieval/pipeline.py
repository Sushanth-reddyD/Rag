"""Multi-stage retrieval pipeline with re-ranking."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    
    # Number of candidates to retrieve at each stage
    initial_k: int = 50
    final_k: int = 5
    
    # Retrieval methods to use
    use_dense: bool = True
    use_sparse: bool = True
    
    # Re-ranking
    use_reranking: bool = True
    rerank_top_k: int = 20
    
    # Scoring weights
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    # Thresholds
    min_score_threshold: float = 0.3
    
    # Query processing
    expand_query: bool = True
    normalize_query: bool = True


class QueryProcessor:
    """Process and normalize queries."""
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query text.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters (keep alphanumeric and spaces)
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        return normalized
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Expand query with variations.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        queries = [query]
        
        # Add question variations
        if not query.endswith('?'):
            queries.append(f"{query}?")
        
        # Add "what is" variation if not present
        if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            queries.append(f"what is {query}")
        
        return queries


class SparseRetriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize sparse retriever.
        
        Args:
            documents: Optional list of documents to index
        """
        self.documents = documents or []
        self.bm25 = None
        self._tokenized_corpus = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of document dicts with 'text' field
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for sparse retrieval. "
                "Install with: pip install rank-bm25"
            )
        
        self.documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc.get('text', ''))
            for doc in documents
        ]
        
        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search documents using BM25.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        # Return documents with normalized scores
        max_score = max(scores) if max(scores) > 0 else 1.0
        results = [
            (self.documents[i], float(scores[i]) / max_score)
            for i in top_k_indices
        ]
        
        return results


class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline.
    Combines dense vector search with sparse lexical search.
    """
    
    def __init__(
        self,
        vector_store,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            vector_store: VectorStore or HybridVectorStore instance
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.query_processor = QueryProcessor()
        self.sparse_retriever = None
    
    def initialize_sparse_retrieval(self) -> None:
        """
        Initialize sparse retrieval with documents from vector store.
        This should be called after documents are added to vector store.
        """
        if not self.config.use_sparse:
            return
        
        # Note: In a full implementation, we would load documents from
        # the vector store. For now, this is a placeholder.
        self.sparse_retriever = SparseRetriever()
        print("Sparse retrieval initialized (requires document corpus)")
    
    def _dense_retrieval(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform dense vector retrieval.
        
        Args:
            query: Query text
            k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (text, score, metadata) tuples
        """
        return self.vector_store.similarity_search(
            query, k, filter_metadata
        )
    
    def _sparse_retrieval(
        self,
        query: str,
        k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform sparse lexical retrieval.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (text, score, metadata) tuples
        """
        if not self.sparse_retriever or not self.sparse_retriever.documents:
            return []
        
        results = self.sparse_retriever.search(query, k)
        
        # Convert to consistent format
        formatted = [
            (doc.get('text', ''), score, doc.get('metadata', {}))
            for doc, score in results
        ]
        
        return formatted
    
    def _hybrid_fusion(
        self,
        dense_results: List[Tuple[str, float, Dict[str, Any]]],
        sparse_results: List[Tuple[str, float, Dict[str, Any]]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Fuse dense and sparse results using weighted combination.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Fused and sorted results
        """
        # Create a dict to combine scores by text/id
        combined = {}
        
        # Add dense results
        for text, score, metadata in dense_results:
            key = metadata.get('chunk_id', text[:100])
            combined[key] = {
                'text': text,
                'metadata': metadata,
                'dense_score': score,
                'sparse_score': 0.0
            }
        
        # Add sparse results
        for text, score, metadata in sparse_results:
            key = metadata.get('chunk_id', text[:100])
            if key in combined:
                combined[key]['sparse_score'] = score
            else:
                combined[key] = {
                    'text': text,
                    'metadata': metadata,
                    'dense_score': 0.0,
                    'sparse_score': score
                }
        
        # Compute weighted scores
        fused_results = []
        for key, data in combined.items():
            fused_score = (
                self.config.dense_weight * data['dense_score'] +
                self.config.sparse_weight * data['sparse_score']
            )
            fused_results.append((
                data['text'],
                fused_score,
                data['metadata']
            ))
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Re-rank results using cross-encoder (simplified version).
        
        Args:
            query: Original query
            results: Results to re-rank
            k: Number of top results to return
            
        Returns:
            Re-ranked results
        """
        # Simplified re-ranking: boost results with exact keyword matches
        query_tokens = set(query.lower().split())
        
        reranked = []
        for text, score, metadata in results:
            text_tokens = set(text.lower().split())
            
            # Calculate keyword overlap
            overlap = len(query_tokens & text_tokens)
            keyword_boost = overlap / max(len(query_tokens), 1)
            
            # Combine original score with keyword boost
            new_score = 0.7 * score + 0.3 * keyword_boost
            
            reranked.append((text, new_score, metadata))
        
        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:k]
    
    def _post_filter(
        self,
        results: List[Tuple[str, float, Dict[str, Any]]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Apply post-filtering to results.
        
        Args:
            results: Results to filter
            
        Returns:
            Filtered results
        """
        # Filter by minimum score threshold
        filtered = [
            (text, score, metadata)
            for text, score, metadata in results
            if score >= self.config.min_score_threshold
        ]
        
        # Remove near-duplicates based on text similarity
        deduplicated = []
        seen_texts = set()
        
        for text, score, metadata in filtered:
            # Use first 200 chars as deduplication key
            text_key = text[:200].lower()
            
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduplicated.append((text, score, metadata))
        
        return deduplicated
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method implementing the full pipeline.
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieval results with metadata
        """
        start_time = time.time()
        
        # Step 1: Query normalization
        if self.config.normalize_query:
            normalized_query = self.query_processor.normalize_query(query)
        else:
            normalized_query = query
        
        # Step 2: Query expansion (optional)
        queries = [normalized_query]
        if self.config.expand_query:
            queries.extend(self.query_processor.expand_query(normalized_query))
        
        # Step 3: Initial retrieval
        all_results = []
        
        for q in queries[:2]:  # Limit to 2 query variations
            # Dense retrieval
            if self.config.use_dense:
                dense_results = self._dense_retrieval(
                    q,
                    self.config.initial_k,
                    filter_metadata
                )
                all_results.extend(dense_results)
            
            # Sparse retrieval
            if self.config.use_sparse and self.sparse_retriever:
                sparse_results = self._sparse_retrieval(
                    q,
                    self.config.initial_k
                )
                all_results.extend(sparse_results)
        
        # Step 4: Fusion (if using hybrid)
        if self.config.use_dense and self.config.use_sparse:
            # For simplicity, we just deduplicate here
            # In full implementation, we'd use proper fusion
            pass
        
        # Remove duplicates
        unique_results = []
        seen_ids = set()
        for text, score, metadata in all_results:
            chunk_id = metadata.get('chunk_id', text[:50])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append((text, score, metadata))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Re-ranking
        if self.config.use_reranking:
            reranked = self._rerank_results(
                query,
                unique_results[:self.config.rerank_top_k],
                self.config.final_k * 2  # Get more for filtering
            )
        else:
            reranked = unique_results[:self.config.final_k * 2]
        
        # Step 6: Post-filtering
        filtered = self._post_filter(reranked)
        
        # Step 7: Format final results
        final_results = []
        for rank, (text, score, metadata) in enumerate(filtered[:self.config.final_k]):
            final_results.append({
                'rank': rank + 1,
                'text': text,
                'score': score,
                'metadata': metadata,
                'retrieval_method': 'hybrid' if self.config.use_sparse else 'dense',
                'confidence': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'low'
            })
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'results': final_results,
            'query': query,
            'normalized_query': normalized_query,
            'num_results': len(final_results),
            'retrieval_latency_ms': latency_ms
        }
