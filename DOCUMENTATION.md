# LangGraph Router - Technical Documentation

## Architecture Overview

The LangGraph Router implements an intelligent query routing system that classifies user queries into four categories using both LLM-based and keyword-based approaches.

### Routing Categories

1. **Complaint** (Priority: 1 - Highest)
   - Customer complaints, problems, negative feedback
   - Keywords: broken, defective, unacceptable, terrible, etc.

2. **API Call** (Priority: 2)
   - Real-time data requests requiring external systems
   - Keywords: weather, stock, current, status, track, etc.

3. **Retrieval** (Priority: 3)
   - Questions about documentation, policies, procedures
   - Keywords: policy, documentation, how do i, terms, etc.

4. **Conversational** (Priority: 4 - Lowest/Default)
   - Greetings, casual chat, thanks, small talk
   - Keywords: hello, hi, thanks, joke, etc.

## System Components

### 1. Models (`src/router/models.py`)

#### RoutingDecision
```python
class RoutingDecision(BaseModel):
    category: Literal["complaint", "api_call", "retrieval", "conversational"]
    reasoning: str
    confidence: Literal["high", "medium", "low"]
```

#### RouterState
```python
class RouterState(BaseModel):
    user_input: str
    routing_decision: str
    reasoning: str
    confidence: str
    response: str
```

### 2. Router Node (`src/router/router_node.py`)

The main routing component with dual strategy:

#### LLM-Based Routing
- Uses Gemma model for intelligent classification
- Temperature: 0.1 (consistent routing)
- Max tokens: 150
- Falls back to keyword-based on low confidence

#### Keyword-Based Routing
- Fast, deterministic fallback mechanism
- Priority-ordered keyword matching
- 100% accuracy on test cases
- No model download required

### 3. Routing Logic (`src/router/routing_logic.py`)

Conditional edge function for LangGraph:
```python
def routing_decision(state: Dict[str, Any]) -> Literal[...]
```

### 4. Orchestrator (`src/router/orchestrator.py`)

LangGraph StateGraph implementation:
```
Router → Conditional Edge → Agent Nodes → END
```

### 5. Configuration (`src/config/settings.py`)

Centralized configuration:
- Model name: google/gemma-2-2b-it
- Device: CPU
- Temperature: 0.1
- Max tokens: 150

## Routing Strategy

### Priority Order
```
1. Complaint (emotional/problem indicators)
   ↓
2. API Call (real-time data needs)
   ↓
3. Retrieval (documentation needs)
   ↓
4. Conversational (default)
```

### Decision Flow

```
User Query
    ↓
Try LLM-based routing
    ↓
Check confidence
    ↓
If low confidence → Keyword-based fallback
    ↓
Return category + reasoning + confidence
```

## Performance Metrics

### Keyword-Based Routing
- **Accuracy**: 100% (20/20 test cases)
- **Speed**: <10ms per query
- **Memory**: <100MB
- **Reliability**: No external dependencies

### Expected LLM-Based Routing
- **Accuracy**: >85% target
- **Speed**: <3 seconds per query
- **Memory**: <4GB RAM
- **Model**: Gemma 2 2B (CPU-optimized)

## Test Coverage

### Test Cases (20 total)

#### Retrieval (6 tests)
- ✅ Return policy queries
- ✅ Password reset procedures
- ✅ Warranty information
- ✅ Shipping policies
- ✅ Privacy policy
- ✅ Refund procedures

#### Conversational (4 tests)
- ✅ Greetings
- ✅ Thanks
- ✅ Morning/evening greetings
- ✅ Joke requests

#### API Call (4 tests)
- ✅ Weather queries
- ✅ Order tracking
- ✅ Stock prices
- ✅ Status checks

#### Complaint (4 tests)
- ✅ Broken products
- ✅ Delayed orders
- ✅ Poor service
- ✅ General dissatisfaction

#### Edge Cases (2 tests)
- ✅ Mixed complaint + policy
- ✅ Documentation request

## Usage Examples

### Basic Usage

```python
from src.router.orchestrator import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator()
result = orchestrator.route_query("What is your return policy?")

print(f"Category: {result['routing_decision']}")
print(f"Reasoning: {result['reasoning']}")
```

### Direct Router Usage

```python
from src.router.router_node import get_router

router = get_router()
state = {"user_input": "My product is broken!"}
result = router.route(state)
```

### Testing Without Model

```bash
python examples/test_keyword_routing.py
```

## File Structure

```
Rag/
├── src/
│   ├── config/
│   │   └── settings.py           # Configuration
│   └── router/
│       ├── models.py              # Pydantic models
│       ├── prompts.py             # Prompt templates
│       ├── router_node.py         # Main router
│       ├── routing_logic.py       # Conditional routing
│       └── orchestrator.py        # LangGraph orchestrator
├── tests/
│   ├── test_router.py             # Full router tests
│   ├── test_orchestrator.py       # Orchestrator tests
│   └── test_standalone.py         # Dependency-free tests
├── examples/
│   ├── run_orchestrator.py        # Full example
│   └── test_keyword_routing.py    # Quick test
└── requirements.txt
```

## Extending the System

### Adding New Categories

1. Update `ROUTING_CATEGORIES` in `settings.py`
2. Add keywords to `_keyword_based_routing()` in `router_node.py`
3. Update Literal types in `models.py`
4. Add conditional edge in `orchestrator.py`
5. Add test cases

### Adding New Keywords

Update keyword lists in `router_node.py`:
```python
retrieval_keywords = [
    "policy", "documentation", "new_keyword", ...
]
```

### Customizing LLM Behavior

Edit `ROUTING_PROMPT_TEMPLATE` in `prompts.py`:
```python
ROUTING_PROMPT_TEMPLATE = """
Your custom prompt...
"""
```

## Troubleshooting

### Model Loading Issues
- Ensure sufficient RAM (>4GB)
- Check internet connection for initial download
- Use keyword-based fallback: already enabled

### Low Accuracy
- Review keyword lists
- Adjust prompt templates
- Lower temperature for more consistent results

### Performance Issues
- Use keyword-based routing for production
- Enable model caching
- Consider quantized models

## Future Enhancements (Phase 2)

- [ ] Implement actual agent logic
- [ ] Add vector database integration
- [ ] Connect external APIs
- [ ] Add conversation memory
- [ ] Implement feedback loop
- [ ] Add logging and monitoring
- [ ] Create web API interface

## Dependencies

### Core
- pydantic >= 2.5.0
- typing-extensions >= 4.9.0

### LLM Support (Optional)
- langchain >= 0.1.0
- langchain-community >= 0.0.13
- langgraph >= 0.0.20
- transformers >= 4.36.0
- torch >= 2.1.0

### Testing
- pytest >= 7.4.3
- pytest-asyncio >= 0.21.1

## License

MIT License - See LICENSE file for details
