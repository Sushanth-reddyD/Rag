"""Router node implementation using Gemma model."""

import re
from typing import Dict, Any
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from .prompts import ROUTING_PROMPT_TEMPLATE, FALLBACK_ROUTING_PROMPT
from .models import RoutingDecision
from ..config import MODEL_NAME, DEVICE, TEMPERATURE, MAX_TOKENS


class RouterNode:
    """Router node for classifying queries."""
    
    def __init__(self):
        """Initialize the router with Gemma model."""
        self.model = None
        self.tokenizer = None
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load and cache the Gemma model."""
        print(f"🔄 Loading model: {MODEL_NAME} on {DEVICE}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with CPU optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map=DEVICE
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            device=DEVICE,
            return_full_text=False
        )
        
        # Wrap in LangChain
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        print("✅ Model loaded successfully")
    
    def _parse_response(self, response: str) -> RoutingDecision:
        """Parse LLM response into structured output."""
        response = response.strip().lower()
        
        # Extract category
        category = "conversational"  # default fallback
        reasoning = "Unable to determine specific category"
        confidence = "low"
        
        # Try to extract structured format
        category_match = re.search(r'category:\s*(\w+)', response)
        reasoning_match = re.search(r'reasoning:\s*([^\n]+)', response)
        confidence_match = re.search(r'confidence:\s*(\w+)', response)
        
        if category_match:
            extracted_category = category_match.group(1).strip()
            if extracted_category in ["complaint", "api_call", "retrieval", "conversational"]:
                category = extracted_category
        
        # Fallback: direct keyword detection
        if not category_match:
            if any(word in response for word in ["complaint", "unhappy", "problem", "issue"]):
                category = "complaint"
            elif any(word in response for word in ["api_call", "real-time", "current", "live"]):
                category = "api_call"
            elif any(word in response for word in ["retrieval", "documentation", "policy", "docs"]):
                category = "retrieval"
            else:
                category = "conversational"
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        if confidence_match:
            conf = confidence_match.group(1).strip()
            if conf in ["high", "medium", "low"]:
                confidence = conf
        
        return RoutingDecision(
            category=category,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _keyword_based_routing(self, user_input: str) -> RoutingDecision:
        """Fallback keyword-based routing."""
        user_input_lower = user_input.lower()
        
        # Complaint keywords (highest priority)
        complaint_keywords = [
            "broken", "defective", "unacceptable", "angry", "frustrated",
            "complaint", "complain", "unhappy", "disappointed", "terrible",
            "awful", "worst", "hate", "never again", "refund", "damaged"
        ]
        
        # API call keywords
        api_keywords = [
            "weather", "stock", "current", "status", "track", "tracking",
            "order status", "delivery status", "what's the", "check status"
        ]
        
        # Retrieval keywords
        retrieval_keywords = [
            "policy", "documentation", "how do i", "where can i find",
            "what is your", "procedure", "terms", "privacy", "warranty",
            "about us", "faq", "guide", "manual", "instructions"
        ]
        
        # Conversational keywords
        conversational_keywords = [
            "hello", "hi", "hey", "good morning", "good evening",
            "thanks", "thank you", "bye", "goodbye", "joke"
        ]
        
        # Check in priority order
        if any(keyword in user_input_lower for keyword in complaint_keywords):
            return RoutingDecision(
                category="complaint",
                reasoning="Detected complaint-related keywords",
                confidence="high"
            )
        
        if any(keyword in user_input_lower for keyword in api_keywords):
            return RoutingDecision(
                category="api_call",
                reasoning="Detected real-time data request keywords",
                confidence="high"
            )
        
        if any(keyword in user_input_lower for keyword in retrieval_keywords):
            return RoutingDecision(
                category="retrieval",
                reasoning="Detected documentation/policy keywords",
                confidence="high"
            )
        
        if any(keyword in user_input_lower for keyword in conversational_keywords):
            return RoutingDecision(
                category="conversational",
                reasoning="Detected conversational keywords",
                confidence="high"
            )
        
        # Default to conversational
        return RoutingDecision(
            category="conversational",
            reasoning="No specific keywords detected, defaulting to conversational",
            confidence="low"
        )
    
    def route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the user query to appropriate category.
        
        Args:
            state: Dictionary containing 'user_input'
            
        Returns:
            Updated state with routing_decision, reasoning, and confidence
        """
        user_input = state.get("user_input", "")
        
        print(f"\n🎯 Routing query: {user_input}")
        
        try:
            # Try LLM-based routing first
            prompt = ROUTING_PROMPT_TEMPLATE.format(user_input=user_input)
            response = self.llm.invoke(prompt)
            
            print(f"📝 LLM Response: {response[:200]}...")
            
            decision = self._parse_response(response)
            
            # If confidence is low, use keyword-based fallback
            if decision.confidence == "low":
                print("⚠️ Low confidence, using keyword-based fallback")
                decision = self._keyword_based_routing(user_input)
            
        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}, using keyword-based fallback")
            decision = self._keyword_based_routing(user_input)
        
        print(f"✅ Routing Decision: {decision.category}")
        print(f"💡 Reasoning: {decision.reasoning}")
        print(f"🎚️ Confidence: {decision.confidence}")
        
        return {
            "user_input": user_input,
            "routing_decision": decision.category,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence
        }


# Singleton instance
_router_instance = None


def get_router() -> RouterNode:
    """Get or create router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = RouterNode()
    return _router_instance
