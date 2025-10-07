# LangGraph Orchestrator - Phase 1: Router Implementation

A LangGraph-based orchestrator that intelligently routes user queries to specialized agents using the Gemma model on CPU.

## 🎯 Overview

This project implements a query routing system that classifies user inputs into four categories:

1. **Conversational** - Greetings, casual chat, thanks, small talk
2. **Retrieval** - Documentation, policies, procedures queries
3. **API Call** - Real-time data requests (weather, order status)
4. **Complaint** - Customer complaints, problems, negative feedback

## 🏗️ Architecture

```
User Query
    ↓
Router Node (Gemma Model)
    ↓
Conditional Routing
    ↓
├── Complaint Agent
├── API Call Agent
├── Retrieval Agent
└── Conversational Agent
```

## 📁 Project Structure

```
Rag/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration settings
│   └── router/
│       ├── __init__.py
│       ├── models.py             # Pydantic models
│       ├── prompts.py            # Routing prompts
│       ├── router_node.py        # Main router logic
│       ├── routing_logic.py      # Conditional routing
│       └── orchestrator.py       # LangGraph orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_router.py           # Router tests
│   └── test_orchestrator.py     # Orchestrator tests
├── examples/
│   └── run_orchestrator.py      # Example usage
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Sushanth-reddyD/Rag.git
cd Rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from src.router.orchestrator import LangGraphOrchestrator

# Initialize orchestrator
orchestrator = LangGraphOrchestrator()

# Route a query
result = orchestrator.route_query("What is your return policy?")

print(f"Category: {result['routing_decision']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Confidence: {result['confidence']}")
```

### Run Examples

```bash
python examples/run_orchestrator.py
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_router.py -v
pytest tests/test_orchestrator.py -v
```

## 📊 Test Cases

### Retrieval Queries
- "What is your return policy?"
- "How do I reset my password?"
- "Where can I find the warranty information?"
- "What are your shipping policies?"

### Conversational Queries
- "Hello, how are you?"
- "Thanks for your help!"

### API Call Queries
- "What's the weather in London?"
- "Track my order #12345"

### Complaint Queries
- "My product arrived broken!"
- "I've been waiting for 2 weeks, this is unacceptable!"

### Edge Cases
- "Can you help me understand your return policy? Mine is defective." → complaint
- "I need the documentation for returns" → retrieval

## 🎯 Success Metrics

- ✅ Routing accuracy: >85% on test cases
- ✅ Average routing time: <3 seconds
- ✅ Memory usage: <4GB RAM
- ✅ CPU-optimized model loading

## 🔧 Configuration

Edit `src/config/settings.py` to customize:

```python
MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cpu"
TEMPERATURE = 0.1
MAX_TOKENS = 150
```

## 🛠️ Features

### Routing Strategy

1. **LLM-based routing** - Uses Gemma model for intelligent classification
2. **Keyword-based fallback** - Falls back to keyword detection if LLM fails
3. **Structured output** - Uses Pydantic models for type safety
4. **Priority-based routing** - complaint > api_call > retrieval > conversational

### Key Components

- **RouterNode**: Handles model loading and query classification
- **LangGraphOrchestrator**: Manages the workflow graph
- **Conditional Routing**: Routes queries to appropriate agents
- **Placeholder Agents**: Demonstrates routing without full implementation

## 📝 Phase 1 Scope

✅ Router implementation with Gemma model  
✅ Structured output with Pydantic models  
✅ Conditional routing logic  
✅ Comprehensive test suite  
✅ Example usage scripts  
❌ Agent implementations (Phase 2)  
❌ Vector database integration (Phase 2)  

## 🔄 Next Steps (Phase 2)

- Implement actual agent logic
- Add vector database for retrieval agent
- Integrate external APIs for API call agent
- Add conversation memory
- Implement feedback loop

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.