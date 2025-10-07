"""Router node implementation using BERT model."""

import re
from typing import Dict, Any
import numpy as np
import os
import json
import time
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

from .prompts import ROUTING_PROMPT_TEMPLATE, FALLBACK_ROUTING_PROMPT
from .models import RoutingDecision
from ..config import MODEL_NAME, DEVICE, TEMPERATURE, MAX_TOKENS, USE_FINE_TUNED, FINE_TUNED_MODEL_PATH


class RouterNode:
    """Router node for classifying queries."""
    
    def __init__(self):
        """Initialize the router with BERT model."""
        self.model = None
        self.tokenizer = None
        self.category_embeddings = None
        self.use_fine_tuned = USE_FINE_TUNED
        self.id_to_category = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load and cache the BERT model."""
        
        if self.use_fine_tuned and os.path.exists(FINE_TUNED_MODEL_PATH):
            # Load fine-tuned model
            print(f"🔄 Loading fine-tuned model from {FINE_TUNED_MODEL_PATH} on {DEVICE}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
            
            # Load label mappings
            with open(f"{FINE_TUNED_MODEL_PATH}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
            
            # Convert string keys back to integers for id_to_category
            self.id_to_category = {int(k): v for k, v in mappings['id_to_category'].items()}
            
            print("✅ Fine-tuned BERT model loaded successfully")
            
        else:
            # Load base model for similarity-based routing
            if self.use_fine_tuned:
                print(f"⚠️ Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}")
                print("📌 Falling back to base BERT with similarity routing")
            
            print(f"🔄 Loading base model: {MODEL_NAME} on {DEVICE}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME)
            
            print("✅ Base BERT model loaded successfully")
            
            # Pre-compute embeddings for category descriptions
            self._compute_category_embeddings()
        
        # Set model to evaluation mode
        self.model.eval()
        
        if DEVICE == "cuda":
            self.model = self.model.to(DEVICE)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for a text."""
        # Tokenize and get model output
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding.cpu().numpy()
    
    def _compute_category_embeddings(self):
        """Pre-compute embeddings for category descriptions."""
        from ..config import ROUTING_CATEGORIES
        
        print("🔄 Computing category embeddings...")
        
        self.category_embeddings = {}
        
        # Create multiple embeddings per category and average them for better representation
        category_examples = {
            "complaint": [
                "My product is broken and defective",
                "I am very angry and frustrated with this terrible service",
                "This is unacceptable, I want a refund for my damaged item",
                "Worst experience ever, completely disappointed",
                "This product arrived broken, I'm unhappy"
            ],
            "api_call": [
                "What is the current weather forecast right now",
                "Track my order status at this moment",
                "Check the live stock prices today",
                "Where is my delivery currently located",
                "Get real-time updates on shipping"
            ],
            "retrieval": [
                "What is your company return policy documentation",
                "Where can I find information about shipping procedures",
                "Tell me about your warranty terms and conditions",
                "How do I access the user manual and FAQ guides",
                "Explain your privacy policy and terms"
            ],
            "conversational": [
                "Hello how are you doing today",
                "Thank you so much for all your help",
                "Good morning have a wonderful day",
                "Goodbye see you later bye"
            ]
        }
        
        for category, examples in category_examples.items():
            embeddings = []
            for example in examples:
                embedding = self._get_embedding(example)
                embeddings.append(embedding)
            
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            self.category_embeddings[category] = avg_embedding
        
        print("✅ Category embeddings computed")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _bert_based_routing(self, user_input: str) -> RoutingDecision:
        """Use BERT embeddings for semantic routing."""
        # Get embedding for user query
        query_embedding = self._get_embedding(user_input)
        
        # Calculate similarity with each category
        similarities = {}
        for category, cat_embedding in self.category_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, cat_embedding)
            similarities[category] = similarity
        
        # Get best match
        best_category = max(similarities, key=similarities.get)
        best_score = similarities[best_category]
        
        # Determine confidence based on score difference
        sorted_scores = sorted(similarities.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_diff = sorted_scores[0] - sorted_scores[1]
            if score_diff > 0.1:
                confidence = "high"
            elif score_diff > 0.05:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "medium"
        
        # Override confidence if score is very high
        if best_score > 0.8:
            confidence = "high"
        elif best_score < 0.5:
            confidence = "low"
        
        reasoning = f"BERT semantic similarity: {best_score:.3f} (scores: " + ", ".join([f"{k}={v:.3f}" for k, v in sorted(similarities.items(), key=lambda x: x[1], reverse=True)]) + ")"
        
        return RoutingDecision(
            category=best_category,
            reasoning=reasoning[:200],  # Truncate to fit model
            confidence=confidence
        )
    
    def _fine_tuned_classification(self, user_input: str) -> RoutingDecision:
        """Use fine-tuned BERT for classification."""
        # Tokenize input
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence_score = probs[0][pred_id].item()
        category = self.id_to_category[pred_id]
        
        # Determine confidence level
        if confidence_score > 0.8:
            confidence = "high"
        elif confidence_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Get top predictions for reasoning
        top_probs, top_ids = torch.topk(probs[0], min(3, len(self.id_to_category)))
        reasoning_parts = [
            f"{self.id_to_category[idx.item()]}={prob.item():.3f}"
            for prob, idx in zip(top_probs, top_ids)
        ]
        reasoning = f"Fine-tuned BERT: {', '.join(reasoning_parts)}"
        
        return RoutingDecision(
            category=category,
            reasoning=reasoning[:200],
            confidence=confidence
        )
    
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
        
        # Complaint keywords (highest priority) - focus on emotional/problem language
        complaint_keywords = [
            "broken", "defective", "unacceptable", "angry", "frustrated",
            "complaint", "complain", "unhappy", "disappointed", "terrible",
            "awful", "worst", "hate", "never again", "damaged"
        ]
        
        # API call keywords
        api_keywords = [
            "weather", "stock", "current", "status", "track", "tracking",
            "order status", "delivery status", "what's the", "check status"
        ]
        
        # Retrieval keywords - includes procedural questions
        retrieval_keywords = [
            "policy", "policies", "documentation", "how do i", "where can i find",
            "what is your", "what are your", "procedure", "terms", "privacy", "warranty",
            "about us", "faq", "guide", "manual", "instructions", "shipping", "submit"
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
        
        # Start timing
        router_start = time.time()
        
        try:
            # Use fine-tuned model if available, otherwise use similarity-based routing
            if self.use_fine_tuned and self.id_to_category is not None:
                print("🧠 Using fine-tuned BERT classifier...")
                decision = self._fine_tuned_classification(user_input)
                
                # For fine-tuned model, we trust it more but still check for edge cases
                if decision.confidence == "low":
                    print("⚠️ Low confidence from fine-tuned model, checking keywords...")
                    keyword_decision = self._keyword_based_routing(user_input)
                    if keyword_decision.confidence == "high":
                        print(f"✅ Using keyword override: {keyword_decision.category}")
                        decision = keyword_decision
            
            else:
                # Use BERT semantic similarity for routing
                print("🧠 Using BERT semantic similarity for routing...")
                decision = self._bert_based_routing(user_input)
                
                # Check for strong keyword signals that might override BERT
                keyword_decision = self._keyword_based_routing(user_input)
                
                user_lower = user_input.lower()
                
                # For policy questions ("can i", "how do i", "what is/are your"), prefer retrieval
                policy_patterns = ["can i", "how do i", "what is your", "what are your", "documentation", "need the doc"]
                if any(pattern in user_lower for pattern in policy_patterns):
                    # Check if it's not a strong complaint (no emotion words)
                    strong_complaint_words = ["broken", "defective", "unacceptable", "angry", "frustrated", "terrible", "awful", "worst"]
                    has_complaint_emotion = any(word in user_lower for word in strong_complaint_words)
                    
                    if not has_complaint_emotion:
                        print(f"✅ Policy question pattern detected, routing to: retrieval")
                        decision = RoutingDecision(
                            category="retrieval",
                            reasoning="Question pattern indicates documentation/policy request",
                            confidence="high"
                        )
                    # If it has both policy question AND complaint emotion, keep it as complaint
                    elif has_complaint_emotion and decision.category == "complaint":
                        print(f"✅ Policy question with complaint emotion, keeping as: complaint")
                        # Keep the current complaint decision
                    else:
                        print(f"✅ Policy question pattern detected, routing to: retrieval")
                        decision = RoutingDecision(
                            category="retrieval",
                            reasoning="Question pattern indicates documentation/policy request",
                            confidence="high"
                        )
                # For strong complaint keywords without policy questions, override BERT
                elif keyword_decision.category == "complaint" and keyword_decision.confidence == "high":
                    # Strong complaint keywords should override
                    strong_complaint_words = ["broken", "defective", "unacceptable", "angry", "frustrated", "terrible", "awful", "worst"]
                    if any(word in user_lower for word in strong_complaint_words):
                        print(f"✅ Strong complaint signal, overriding to: complaint")
                        decision = keyword_decision
                # If keyword has very high confidence and BERT is uncertain, use keyword
                elif keyword_decision.confidence == "high" and decision.confidence != "high":
                    print(f"✅ Using keyword-based override: {keyword_decision.category}")
                    decision = keyword_decision
                # If BERT has low confidence, always use keywords
                elif decision.confidence == "low":
                    print(f"✅ BERT low confidence, using keyword decision: {keyword_decision.category}")
                    decision = keyword_decision
            
        except Exception as e:
            print(f"⚠️ Routing failed: {e}, using keyword-based fallback")
            decision = self._keyword_based_routing(user_input)
        
        router_end = time.time()
        router_time = (router_end - router_start) * 1000  # Convert to ms
        
        print(f"✅ Routing Decision: {decision.category}")
        print(f"💡 Reasoning: {decision.reasoning}")
        print(f"🎚️ Confidence: {decision.confidence}")
        print(f"⏱️  Router Time: {router_time:.2f} ms")
        
        # Get existing timing dict or create new one
        timing = state.get('_timing', {})
        timing['router_time'] = router_time
        
        return {
            "user_input": user_input,
            "routing_decision": decision.category,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "_timing": timing
        }


# Singleton instance
_router_instance = None


def get_router() -> RouterNode:
    """Get or create router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = RouterNode()
    return _router_instance
