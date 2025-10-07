# LangGraph Orchestrator - Phase 1: Router Implementation

A LangGraph-based orchestrator that intelligently routes user queries to specialized agents using the Gemma model on CPU.

## 🎯 Overview

This project implements a query routing system that classifies user inputs into four categories:

1. **Conversational** - Greetings, casual chat, thanks, small talk
2. **Retrieval** - Documentation, policies, procedures queries
3. **API Call** - Real-time data requests (weather, order status)
4. **Complaint** - Customer complaints, problems, negative feedback

## 🏗️ Architecture

### Phase 1: Router System
```
User Query
    ↓
Router Node (Gemma Model / Keyword-based)
    ↓
Conditional Routing
    ↓
├── Complaint Agent
├── API Call Agent
├── Retrieval Agent (Phase 2 ✅)
└── Conversational Agent
```

### Phase 2: Retrieval System (NEW)
```
Documents → Ingestion Pipeline → Vector DB → Retrieval Agent
              ↓                      ↓
         [Parse, Chunk,         [ChromaDB +
          Embed, Enrich]         Metadata Store]
                                     ↓
                            Multi-stage Retrieval
                                     ↓
                            Results with Citations
```

**Phase 2 Components:**
- 📄 Document preprocessing (PDF, HTML, DOCX, TXT)
- ✂️ Semantic chunking with overlap
- 🧠 CPU-optimized embeddings (Sentence Transformers)
- 💾 Hybrid storage (ChromaDB + JSON metadata)
- 🔍 Multi-stage retrieval (dense + sparse + re-ranking)
- 📚 Citation and provenance tracking

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

2. Install Phase 1 dependencies (router only):
```bash
pip install -r requirements.txt
```

3. Install Phase 2 dependencies (optional - for vector database):
```bash
pip install -r requirements-phase2.txt
```

**Note:** Phase 2 requires additional dependencies for document processing and vector storage. Phase 1 works standalone with just the core dependencies.

## 💻 Usage

### Basic Usage (Phase 1 - Router Only)

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

### Phase 2 Usage (With Vector Database)

```python
# 1. Ingest documents
from src.vectordb.store import VectorStore, HybridVectorStore
from src.ingestion.pipeline import IngestionPipeline

vector_store = VectorStore(collection_name="my_docs")
hybrid_store = HybridVectorStore(vector_store)
pipeline = IngestionPipeline(hybrid_store, chunking_strategy="semantic")

# Ingest a directory
record = pipeline.ingest_directory("./data/documents")
print(f"Ingested {record.total_chunks} chunks from {record.processed_documents} documents")

# 2. Use orchestrator with real retrieval
orchestrator = LangGraphOrchestrator(use_real_retrieval=True)
result = orchestrator.route_query("What is your return policy?")
print(result['response'])  # Gets actual retrieved content with citations
```

### Run Examples

```bash
# Phase 1 examples
python examples/run_orchestrator.py
python examples/test_keyword_routing.py

# Phase 2 examples
python examples/demo_phase2.py                    # Full ingestion & retrieval demo
python examples/run_orchestrator_phase2.py        # Integrated system demo
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

## 📝 Phase 2 Scope (Completed)

✅ Document ingestion pipeline (PDF, HTML, DOCX, TXT)  
✅ Multiple chunking strategies (Fixed, Sentence, Semantic)  
✅ Vector database integration (ChromaDB - CPU optimized)  
✅ Hybrid storage (Vector + Metadata)  
✅ Multi-stage retrieval pipeline  
✅ Re-ranking and score fusion  
✅ Retrieval agent with citation tracking  
✅ Comprehensive Phase 2 test suite  
✅ Full integration with Phase 1 router  

## 🔄 Next Steps (Phase 3)

- Implement API call agent with external integrations
- Implement complaint handling agent
- Add conversation memory and context tracking
- Implement feedback loop for continuous improvement
- Create web API interface
- Add multilingual support

## 📄 License

This project is open source and available under the MIT License.

## 📖 Documentation

- **README.md** - Main project overview
- **PHASE1_SUMMARY.md** - Phase 1 implementation details
- **PHASE2_DOCUMENTATION.md** - Phase 2 comprehensive guide (NEW)
- **DOCUMENTATION.md** - Technical documentation
- **QUICKSTART.md** - Quick start guide
- **FINE_TUNING_SUMMARY.md** - Router fine-tuning guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.