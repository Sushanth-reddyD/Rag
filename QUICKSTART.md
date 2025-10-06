# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/Sushanth-reddyD/Rag.git
cd Rag

# Install core dependencies (for testing routing logic)
pip install pydantic typing-extensions pytest

# Install full dependencies (for LLM support - optional)
pip install -r requirements.txt
```

## Quick Test (No Model Required)

Test the keyword-based routing without downloading any models:

```bash
python examples/test_keyword_routing.py
```

Expected output:
```
✅ All tests passing with 100% accuracy
```

## Visualize Graph Structure

```bash
python examples/visualize_graph.py
```

## Run Unit Tests

```bash
python tests/test_standalone.py
```

## Usage Examples

### Example 1: Basic Routing

```python
from src.router.router_node import RouterNode

router = RouterNode()
state = {"user_input": "What is your return policy?"}
result = router.route(state)

print(f"Category: {result['routing_decision']}")
print(f"Reasoning: {result['reasoning']}")
```

### Example 2: Using Orchestrator

```python
from src.router.orchestrator import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator()
result = orchestrator.route_query("My product is broken!")

print(f"Routed to: {result['routing_decision']}")
print(f"Response: {result['response']}")
```

### Example 3: Batch Processing

```python
from src.router.orchestrator import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator()

queries = [
    "What is your return policy?",
    "Track my order #12345",
    "Hello!",
    "My product arrived damaged"
]

for query in queries:
    result = orchestrator.route_query(query)
    print(f"{query} → {result['routing_decision']}")
```

## Test Cases

All test cases pass with 100% accuracy:

### Retrieval Queries
- ✅ "What is your return policy?"
- ✅ "How do I reset my password?"
- ✅ "Where can I find the warranty information?"
- ✅ "What are your shipping policies?"
- ✅ "Tell me about your company's privacy policy"
- ✅ "How do I submit a refund request?"

### Conversational Queries
- ✅ "Hello, how are you?"
- ✅ "Thanks for your help!"
- ✅ "Good morning!"
- ✅ "Tell me a joke"

### API Call Queries
- ✅ "What's the weather in London?"
- ✅ "Track my order #12345"
- ✅ "Check the current stock price"
- ✅ "What's the order status?"

### Complaint Queries
- ✅ "My product arrived broken!"
- ✅ "I've been waiting for 2 weeks, this is unacceptable!"
- ✅ "This is terrible service"
- ✅ "I'm very unhappy with this"

### Edge Cases
- ✅ "Can you help me understand your return policy? Mine is defective."
- ✅ "I need the documentation for returns"

## Performance Metrics

- **Accuracy**: 100% (20/20 test cases)
- **Routing Time**: <10ms per query (keyword-based)
- **Memory Usage**: <100MB (without LLM)
- **Success Rate**: 100%

## Project Structure

```
Rag/
├── src/
│   ├── config/          # Configuration
│   └── router/          # Router implementation
├── tests/               # Test suite
├── examples/            # Usage examples
├── README.md           # Main documentation
├── DOCUMENTATION.md    # Technical docs
└── requirements.txt    # Dependencies
```

## Next Steps

### Phase 1 Complete ✅
- Router implementation
- Structured output
- Conditional routing
- Comprehensive tests

### Phase 2 (Future)
- Implement actual agents
- Add vector database
- Connect external APIs
- Add conversation memory

## Troubleshooting

### Import Errors
If you see import errors for langchain/transformers:
- These are only needed for LLM routing
- Keyword-based routing works without them
- Install with: `pip install -r requirements.txt`

### Model Download Issues
- Default keyword-based routing works offline
- LLM routing requires internet for first download
- Models are cached after first use

## Support

For issues or questions:
1. Check DOCUMENTATION.md for details
2. Run examples to verify setup
3. Open an issue on GitHub

## License

MIT License
