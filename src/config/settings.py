"""Configuration settings for LangGraph Router."""

# Model configuration
MODEL_NAME = "google/gemma-2-2b-it"  # Using Gemma 2 2B as it's more accessible than 270M
DEVICE = "cpu"
TEMPERATURE = 0.1
MAX_TOKENS = 150

# Routing categories
ROUTING_CATEGORIES = {
    "complaint": "Customer complaints, problems, negative feedback, or dissatisfaction",
    "api_call": "Real-time data requests requiring external systems",
    "retrieval": "Questions about company documentation, policies, or static information",
    "conversational": "Greetings, casual chat, thanks, small talk"
}

# Priority order (highest to lowest)
ROUTING_PRIORITY = ["complaint", "api_call", "retrieval", "conversational"]
