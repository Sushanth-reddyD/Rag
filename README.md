# Customer Support RAG System

> **Intelligent Customer Service System with Query Routing, Vector Retrieval, and AI Generation**

A production-ready Retrieval-Augmented Generation (RAG) system that intelligently routes customer queries, retrieves relevant information from documents, and generates natural language answers using local or cloud-based AI models.

## 🎯 Overview

This system provides an intelligent customer service solution with:

- **Smart Query Routing** - BERT-based classification routes queries to appropriate handlers
- **Vector Retrieval** - Semantic search across documents using embeddings
- **AI Answer Generation** - Natural language responses using Gemma (local) or Gemini (cloud)
- **Web UI** - Beautiful dark-themed Streamlit interface
- **Production-Ready** - Comprehensive error handling, logging, and performance tracking

## ✨ Key Features

### 1. Intelligent Router
- **Fine-tuned BERT classifier** for accurate query categorization
- **Four routing categories**: Retrieval, Complaint, API Call, Conversational
- **85.7% classification accuracy** on test set
- **Confidence scoring** and reasoning for transparency

### 2. Vector Retrieval System
- **Semantic search** using sentence-transformers (MiniLM-L6-v2)
- **Hybrid storage** with ChromaDB vector store + JSON metadata
- **Multi-stage retrieval** with re-ranking and score fusion
- **Citation tracking** with source attribution
- **Supports multiple formats**: PDF, HTML, DOCX, TXT

### 3. Flexible AI Generation
- **Local option**: Gemma 3 270M (free, private, CPU/GPU)
- **Cloud option**: Gemini 2.5 (fast, powerful, API-based)
- **Structured JSON output** for reliable parsing
- **Detailed timing metrics** for performance optimization

### 4. Modern Web Interface
- **Dark theme** optimized for readability
- **Real-time responses** with streaming support
- **Source citations** displayed cleanly
- **Chat history** maintained during session

## 🏗️ Architecture

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  BERT Router                │
│  • Complaint                │
│  • API Call                 │
│  • Retrieval  ◄─────────────┼───┐
│  • Conversational           │   │
└─────────────────────────────┘   │
                                  │
       ┌──────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Vector Retrieval Pipeline  │
│  1. Query Processing        │
│  2. Embedding Search        │
│  3. Re-ranking              │
│  4. Citation Enrichment     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  AI Generation              │
│  • Gemma 3 270M (local)     │
│  • Gemini 2.5 (cloud)       │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Formatted Response         │
│  • Answer text              │
│  • Source citations         │
│  • Confidence scores        │
└─────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB+ recommended for Gemma)
- Git

### Quick Install

```bash
# Clone repository
git clone https://github.com/Sushanth-reddyD/Rag.git
cd Rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up Gemini API
export GEMINI_API_KEY='your-api-key-here'
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Start web interface
streamlit run app.py
```

## 🚀 Quick Start

### 1. Using the Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start asking questions!

**Example queries:**
- "What is your return policy?"
- "How long do products last?"
- "What is the shipping policy?"

### 2. Using Python API

```python
from src.router.orchestrator import LangGraphOrchestrator

# Initialize system
orchestrator = LangGraphOrchestrator(
    use_real_retrieval=True,
    auto_load_docs=True
)

# Query the system
result = orchestrator.route_query("What is your return policy?")

# Access results
print(result['answer'])          # Generated answer
print(result['routing_decision']) # Which category
print(result['sources'])          # Source documents
```

### 3. Switching Between Gemma and Gemini

**Option A: Environment Variables**
```bash
# Use local Gemma (default)
export MODEL_TYPE='gemma'

# Use cloud Gemini
export MODEL_TYPE='gemini'
export GEMINI_API_KEY='your-key'
```

**Option B: Configuration File**

Edit `src/config/model_config.py`:
```python
MODEL_TYPE = 'gemini'  # or 'gemma'
MODEL_ID = 'gemini-2.5-flash'  # or 'google/gemma-3-270m-it'
GEMINI_API_KEY = 'your-key'  # or read from env
```

## 📚 Configuration

### Model Selection

| Model | Speed | Cost | Privacy | Requirements |
|-------|-------|------|---------|--------------|
| **Gemma 3 270M** | Moderate (10-20 tok/s on CPU) | Free | Local | 4GB RAM |
| **Gemini 2.5 Flash** | Fast (100+ tok/s) | ~$0.001/query | Cloud | API key |
| **Gemini 2.5 Pro** | Fast (80+ tok/s) | ~$0.01/query | Cloud | API key |

### Key Configuration Options

**Router Configuration** (`src/config/model_config.py`):
```python
# Model selection
MODEL_TYPE = 'gemma'  # or 'gemini'
MODEL_ID = 'google/gemma-3-270m-it'

# Generation parameters
GEMMA_CONFIG = {
    'max_context_length': 32000,
    'max_new_tokens': 256,
    'temperature': 0.1,
    'device': 'cpu'  # or 'cuda'
}
```

**Retrieval Configuration** (`src/retrieval/pipeline.py`):
```python
config = RetrievalConfig(
    initial_k=20,        # Initial retrieval size
    final_k=5,           # Final results returned
    score_threshold=0.3, # Minimum relevance score
    use_sparse=False,    # Enable BM25 (optional)
    dense_weight=0.7,    # Dense search weight
    sparse_weight=0.3    # Sparse search weight
)
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Test router
pytest tests/test_router.py -v

# Test retrieval
pytest tests/test_orchestrator.py -v

# Test phase 2 integration
pytest tests/test_phase2.py -v
```

### Test Generation Speed
```bash
python test_gemma_speed.py
```

### Test Embedding Similarity
```bash
python test_embedding_semantic_match.py
```

## 📁 Project Structure

```
Rag/
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
│
├── src/
│   ├── config/
│   │   ├── model_config.py     # Model selection & config
│   │   └── settings.py         # Application settings
│   │
│   ├── router/
│   │   ├── orchestrator.py     # Main orchestration logic
│   │   ├── router_node.py      # BERT routing
│   │   └── routing_logic.py    # Conditional routing
│   │
│   ├── retrieval/
│   │   ├── agent.py            # Retrieval agent
│   │   └── pipeline.py         # Multi-stage retrieval
│   │
│   ├── generation/
│   │   └── model_factory.py    # Gemma & Gemini generators
│   │
│   ├── vectordb/
│   │   └── store.py            # ChromaDB integration
│   │
│   └── ingestion/
│       ├── pipeline.py         # Document ingestion
│       ├── preprocessing.py    # Document parsing
│       └── chunking.py         # Text chunking strategies
│
├── tests/
│   ├── test_router.py          # Router tests
│   ├── test_orchestrator.py    # Integration tests
│   └── test_phase2.py          # Phase 2 tests
│
├── examples/
│   ├── run_orchestrator.py     # Basic usage example
│   └── visualize_graph.py      # Visualize workflow
│
├── data/
│   ├── documents/              # Source documents
│   └── metadata/               # Chunk metadata (JSON)
│
├── models/
│   └── fine_tuned_router/      # Fine-tuned BERT model
│
├── chroma_db/                  # ChromaDB persistence
│
└── docs/
    ├── README.md               # This file
    ├── QUICKSTART.md           # Quick start guide
    └── ARCHITECTURE.md         # Technical architecture
```

## 📊 Performance

### Router Performance
- **Accuracy**: 85.7% on test set
- **Latency**: <500ms per query
- **Memory**: ~1GB (BERT model)

### Retrieval Performance
- **Retrieval time**: 3-5 seconds (initial query)
- **Retrieval time**: <1 second (cached)
- **Precision@5**: ~80% relevant results

### Generation Performance
- **Gemma 3 270M (CPU)**: 10-20 tokens/sec
- **Gemma 3 270M (GPU)**: 40-60 tokens/sec
- **Gemini 2.5 Flash**: 100+ tokens/sec
- **Gemini 2.5 Pro**: 80+ tokens/sec

## 🔧 Troubleshooting

### Common Issues

**1. Import Error: No module named 'src.generation.gemma_generator'**
```bash
# Update to use model_factory instead
from src.generation.model_factory import ModelFactory
```

**2. ChromaDB Connection Error**
```bash
# Delete and rebuild database
rm -rf chroma_db/
python app.py  # Will rebuild automatically
```

**3. Slow Generation on CPU**
```bash
# Switch to Gemini for faster responses
export MODEL_TYPE='gemini'
export GEMINI_API_KEY='your-key'
```

**4. Out of Memory**
```bash
# Reduce context length
# Edit src/config/model_config.py:
GEMMA_CONFIG = {
    'max_context_length': 16000,  # Reduced from 32000
    'max_new_tokens': 128          # Reduced from 256
}
```

## 🚦 Development Roadmap

### Phase 1: Router ✅ (Complete)
- [x] BERT fine-tuning for classification
- [x] Query routing logic
- [x] Test suite
- [x] Performance optimization

### Phase 2: Retrieval ✅ (Complete)
- [x] Document ingestion pipeline
- [x] Vector database integration
- [x] Multi-stage retrieval
- [x] Citation tracking
- [x] Metadata enrichment

### Phase 3: Generation ✅ (Complete)
- [x] Gemma 3 270M integration
- [x] Gemini API integration
- [x] Model factory pattern
- [x] Structured JSON output
- [x] Detailed timing metrics

### Phase 4: UI ✅ (Complete)
- [x] Streamlit web interface
- [x] Dark theme design
- [x] Source citation display
- [x] Chat history

### Phase 5: Future Enhancements
- [ ] Implement Complaint handler agent
- [ ] Implement API Call agent (order tracking, weather)
- [ ] Add conversation memory
- [ ] Multi-turn dialogue support
- [ ] User feedback collection
- [ ] A/B testing framework
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Production deployment guide

## 📖 Documentation

- **[README.md](README.md)** - This file: Project overview and setup
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and design

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Rag.git
cd Rag

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests before committing
pytest tests/
black src/
flake8 src/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sushanth Reddy**
- GitHub: [@Sushanth-reddyD](https://github.com/Sushanth-reddyD)

## 🙏 Acknowledgments

- **Google** - Gemma and Gemini models
- **HuggingFace** - Transformers and model hosting
- **ChromaDB** - Vector database
- **Streamlit** - Web framework


## 📧 Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/Sushanth-reddyD/Rag/issues)
- Check [QUICKSTART.md](QUICKSTART.md) for common solutions
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

---

**Built with ❤️ for better customer service**
