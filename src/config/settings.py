"""Configuration settings for LangGraph Router."""

# Model configuration
MODEL_NAME = "bert-base-uncased"  # Using BERT Base for classification (more memory efficient)
USE_FINE_TUNED = True  # Set to True to use fine-tuned model
FINE_TUNED_MODEL_PATH = "./models/fine_tuned_router"
DEVICE = "cpu"
TEMPERATURE = 0.2
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
