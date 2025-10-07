"""LangGraph orchestrator implementation."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .router_node import get_router
from .routing_logic import routing_decision
from .models import RouterState


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
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.router = get_router()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        workflow = StateGraph(dict)
        
        # Add router node
        workflow.add_node("router", self.router.route)
        
        # Add placeholder agent nodes
        workflow.add_node("complaint_agent", create_placeholder_agent("COMPLAINT"))
        workflow.add_node("api_call_agent", create_placeholder_agent("API CALL"))
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
        initial_state = {
            "user_input": user_input,
            "routing_decision": "",
            "reasoning": "",
            "confidence": "",
            "response": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result
