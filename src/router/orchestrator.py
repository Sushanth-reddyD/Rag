"""LangGraph orchestrator implementation."""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
import time

from .router_node import get_router
from .routing_logic import routing_decision
from .models import RouterState


def auto_load_documents(verbose: bool = False) -> Dict:
    """
    Automatically load new documents on orchestrator initialization
    
    Args:
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        return loader.load_new_documents(verbose=verbose)
    except Exception as e:
        print(f"⚠️ Error loading documents: {str(e)}")
        return {"status": "error", "error": str(e)}


def create_placeholder_agent(agent_name: str):
    """Create a placeholder agent node."""
    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"🤖 {agent_name.upper()} AGENT activated")
        print(f"Query: {state['user_input']}")
        print(f"Routing: {state.get('routing_decision', 'N/A')}")
        print(f"Reasoning: {state.get('reasoning', 'N/A')}")
        print(f"{'='*60}\n")
        
        return {
            **state,
            "response": f"{agent_name} agent would handle: {state['user_input']}"
        }
    return agent_node


class LangGraphOrchestrator:
    """Main orchestrator using LangGraph."""
    
    def __init__(
        self, 
        use_real_retrieval: bool = False, 
        auto_load_docs: bool = True,
        use_gemma_generation: bool = False
    ):
        """
        Initialize the orchestrator.
        
        Args:
            use_real_retrieval: If True, use actual retrieval agent with vector DB
            auto_load_docs: If True, automatically load new documents on startup
            use_gemma_generation: If True, use Gemma 3 270M for answer generation
        """
        # Auto-load documents if enabled
        if auto_load_docs:
            print("🔄 Checking for new documents to load...")
            result = auto_load_documents(verbose=True)
            if result.get("status") == "up_to_date":
                print("✅ All documents up to date\n")
            elif result.get("status") == "success":
                print(f"✅ Loaded {result.get('new_files', 0)} new document(s)\n")
        
        self.router = get_router()
        self.use_real_retrieval = use_real_retrieval
        self.use_gemma_generation = use_gemma_generation
        self.retrieval_agent = None
        self.gemma_generator = None
        
        if use_real_retrieval:
            try:
                from ..retrieval.agent import RetrievalAgent, create_retrieval_agent_node
                self.retrieval_agent = RetrievalAgent()
                print("✅ Real retrieval agent initialized with vector database")
            except ImportError as e:
                print(f"⚠️ Could not initialize real retrieval agent: {e}")
                print("Falling back to placeholder agent")
                self.use_real_retrieval = False
        
        # Initialize Gemma generator if requested
        if use_gemma_generation:
            try:
                from ..generation.model_factory import ModelFactory
                print("🔄 Loading generation model...")
                self.gemma_generator = ModelFactory.create_generator()
                print(f"✅ Generator initialized: {self.gemma_generator.model_id}")
            except ImportError as e:
                print(f"⚠️ Could not initialize generator: {e}")
                print("Install required packages:")
                print("  For Gemma: pip install transformers accelerate")
                print("  For Gemini: pip install google-genai")
                self.use_gemma_generation = False
            except Exception as e:
                print(f"⚠️ Error loading generation model: {e}")
                self.use_gemma_generation = False
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        workflow = StateGraph(dict)
        
        # Add router node
        workflow.add_node("router", self.router.route)
        
        # Add agent nodes
        workflow.add_node("complaint_agent", create_placeholder_agent("COMPLAINT"))
        workflow.add_node("api_call_agent", create_placeholder_agent("API CALL"))
        
        # Add retrieval agent (real or placeholder) with optional Gemma generation
        if self.use_real_retrieval and self.retrieval_agent:
            from ..retrieval.agent import create_retrieval_agent_node
            workflow.add_node(
                "retrieval_agent",
                create_retrieval_agent_node(
                    self.retrieval_agent,
                    gemma_generator=self.gemma_generator if self.use_gemma_generation else None
                )
            )
        else:
            workflow.add_node("retrieval_agent", create_placeholder_agent("RETRIEVAL"))
        
        workflow.add_node("conversational_agent", create_placeholder_agent("CONVERSATIONAL"))
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            routing_decision,
            {
                "complaint": "complaint_agent",
                "api_call": "api_call_agent",
                "retrieval": "retrieval_agent",
                "conversational": "conversational_agent"
            }
        )
        
        # Add edges to END
        workflow.add_edge("complaint_agent", END)
        workflow.add_edge("api_call_agent", END)
        workflow.add_edge("retrieval_agent", END)
        workflow.add_edge("conversational_agent", END)
        
        return workflow.compile()
    
    def route_query(self, user_input: str) -> Dict[str, Any]:
        """
        Route a query through the orchestrator.
        
        Args:
            user_input: The user's query
            
        Returns:
            Final state after routing
        """
        print("\n" + "="*80)
        print("⏱️  TIMING ANALYSIS - STARTING QUERY PROCESSING")
        print("="*80)
        
        total_start = time.time()
        
        initial_state = {
            "user_input": user_input,
            "routing_decision": "",
            "reasoning": "",
            "confidence": "",
            "response": "",
            "_timing": {
                "total_start": total_start
            }
        }
        
        result = self.graph.invoke(initial_state)
        
        total_end = time.time()
        total_time = (total_end - total_start) * 1000  # Convert to ms
        
        # Print timing summary
        print("\n" + "="*80)
        print("⏱️  TIMING SUMMARY")
        print("="*80)
        
        timing = result.get('_timing', {})
        
        if 'router_time' in timing:
            print(f"🔀 Router (BERT):        {timing['router_time']:.2f} ms")
        
        if 'retrieval_time' in timing:
            print(f"🔍 Vector Retrieval:     {timing['retrieval_time']:.2f} ms")
        
        if 'reranking_time' in timing:
            print(f"🎯 Re-ranking:           {timing['reranking_time']:.2f} ms")
        
        if 'generation_time' in timing:
            print(f"🤖 Answer Generation:    {timing['generation_time']:.2f} ms")
            if 'generation_model' in timing:
                print(f"   └─ Model: {timing['generation_model']}")
        
        if 'formatting_time' in timing:
            print(f"✏️  Response Formatting:  {timing['formatting_time']:.2f} ms")
        
        print(f"\n{'─'*80}")
        print(f"⏱️  TOTAL TIME:           {total_time:.2f} ms ({total_time/1000:.2f}s)")
        print("="*80 + "\n")
        
        return result
