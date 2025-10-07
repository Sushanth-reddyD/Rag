"""Retrieval agent with vector database integration."""

from typing import Dict, Any, Optional
from pathlib import Path

from ..vectordb.store import VectorStore, HybridVectorStore
from ..retrieval.pipeline import RetrievalPipeline, RetrievalConfig


class RetrievalAgent:
    """
    Agent that handles retrieval queries using vector database.
    Integrated with the existing router system.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize retrieval agent.
        
        Args:
            collection_name: Name of vector DB collection
            persist_directory: Directory for vector DB persistence
            embedding_model: Embedding model to use
        """
        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        # Initialize hybrid store
        self.hybrid_store = HybridVectorStore(
            vector_store=self.vector_store,
            metadata_store_path="./data/metadata"
        )
        
        # Initialize retrieval pipeline
        self.retrieval_config = RetrievalConfig(
            initial_k=20,
            final_k=5,
            use_dense=True,
            use_sparse=False,  # Can be enabled after sparse indexing
            use_reranking=True,
            min_score_threshold=0.3
        )
        
        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            config=self.retrieval_config
        )
    
    def handle_query(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a retrieval query.
        
        Args:
            query: User query
            state: Current router state
            
        Returns:
            Updated state with retrieval results
        """
        try:
            # Check if vector store has documents
            doc_count = self.vector_store.count()
            
            if doc_count == 0:
                return {
                    **state,
                    "response": (
                        "The knowledge base is currently empty. "
                        "Please ingest documents first using the ingestion pipeline."
                    )
                }
            
            # Perform retrieval
            retrieval_result = self.retrieval_pipeline.retrieve(query)
            
            # Format response with citations
            response = self._format_response(
                query,
                retrieval_result['results']
            )
            
            return {
                **state,
                "response": response,
                "retrieval_results": retrieval_result['results'],
                "retrieval_metadata": {
                    'num_results': retrieval_result['num_results'],
                    'latency_ms': retrieval_result['retrieval_latency_ms'],
                    'total_documents': doc_count
                }
            }
            
        except Exception as e:
            return {
                **state,
                "response": f"Error during retrieval: {str(e)}",
                "error": str(e)
            }
    
    def _format_response(
        self,
        query: str,
        results: list
    ) -> str:
        """
        Format retrieval results into a readable response with citations.
        
        Args:
            query: Original query
            results: List of retrieval results
            
        Returns:
            Formatted response string
        """
        if not results:
            return (
                f"I couldn't find any relevant information about: {query}\n"
                "Please try rephrasing your question or check if the information "
                "has been ingested into the knowledge base."
            )
        
        # Build response with citations
        response_parts = [
            f"Based on the knowledge base, here's what I found about your query:\n"
        ]
        
        for result in results[:3]:  # Show top 3 results
            rank = result['rank']
            text = result['text']
            score = result['score']
            metadata = result['metadata']
            
            # Truncate text if too long
            display_text = text[:300] + "..." if len(text) > 300 else text
            
            # Extract source information
            source = metadata.get('source', 'Unknown source')
            doc_id = metadata.get('document_id', 'unknown')
            chunk_id = metadata.get('chunk_id', 'unknown')
            title = metadata.get('title', Path(source).stem if source else 'Untitled')
            
            response_parts.append(
                f"\n[Result {rank}] (Relevance: {score:.2f})\n"
                f"Source: {title}\n"
                f"{display_text}\n"
                f"(Document ID: {doc_id}, Chunk: {chunk_id})\n"
            )
        
        if len(results) > 3:
            response_parts.append(
                f"\n... and {len(results) - 3} more relevant results."
            )
        
        return "".join(response_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        return {
            'total_documents': self.vector_store.count(),
            'collection_name': self.vector_store.collection_name,
            'embedding_model': self.vector_store.embedding_model_name
        }


def create_retrieval_agent_node(agent: Optional[RetrievalAgent] = None):
    """
    Create a retrieval agent node for LangGraph.
    
    Args:
        agent: Optional RetrievalAgent instance (created if not provided)
        
    Returns:
        Agent node function
    """
    if agent is None:
        agent = RetrievalAgent()
    
    def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieval agent node for LangGraph."""
        query = state.get('user_input', '')
        
        print(f"\n{'='*60}")
        print(f"🔍 RETRIEVAL AGENT activated")
        print(f"Query: {query}")
        print(f"Routing: {state.get('routing_decision', 'N/A')}")
        print(f"Reasoning: {state.get('reasoning', 'N/A')}")
        print(f"{'='*60}\n")
        
        # Handle the query
        result = agent.handle_query(query, state)
        
        # Print results summary
        if 'retrieval_results' in result:
            num_results = len(result['retrieval_results'])
            latency = result.get('retrieval_metadata', {}).get('latency_ms', 0)
            print(f"✅ Retrieved {num_results} results in {latency:.2f}ms\n")
        
        return result
    
    return retrieval_agent_node
