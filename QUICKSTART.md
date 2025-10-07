# Quick Start Guide

> **Get up and running with the Customer Support RAG System in 5 minutes**

## ⚡ Installation

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/Sushanth-reddyD/Rag.git
cd Rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Quick test
pytest tests/test_router.py -v

# Expected output:
# ✅ test_router_initialization PASSED
# ✅ test_routing_accuracy PASSED
```

## 🚀 Your First Query

### Option 1: Web Interface (Recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser and try:
- "What is your return policy?"
- "How long do products last?"
- "What are your shipping policies?"

### Option 2: Python Code

```python
from src.router.orchestrator import LangGraphOrchestrator

# Initialize
orchestrator = LangGraphOrchestrator(use_real_retrieval=True)

# Query
result = orchestrator.route_query("What is your return policy?")

# Results
print(f"Answer: {result['answer']}")
print(f"Category: {result['routing_decision']}")
print(f"Sources: {len(result.get('sources', []))} documents")
```

## 🎯 Choosing Your AI Model

### Gemma (Local - Default)

**Pros**: Free, private, no API key needed
**Cons**: Slower (10-20 tokens/sec on CPU)

```bash
# Already configured by default!
streamlit run app.py
```

### Gemini (Cloud - Recommended for Production)

**Pros**: Fast (100+ tokens/sec), powerful
**Cons**: Requires API key, costs ~$0.001/query

```bash
# 1. Get API key from https://makersuite.google.com/app/apikey
# 2. Set environment variable
export GEMINI_API_KEY='your-api-key-here'

# 3. Configure in src/config/model_config.py
# Change: MODEL_TYPE = 'gemini'

# 4. Run
streamlit run app.py
```

**Quick Switch**:
```bash
# Use Gemma (local)
export MODEL_TYPE='gemma'
python app.py

# Use Gemini (cloud)
export MODEL_TYPE='gemini'
export GEMINI_API_KEY='your-key'
python app.py
```

## 📝 Configuration Guide

### Basic Configuration

Edit `src/config/model_config.py`:

```python
# ===== MODEL SELECTION =====
MODEL_TYPE = 'gemma'  # Options: 'gemma' or 'gemini'

# For Gemma (local)
MODEL_ID = 'google/gemma-3-270m-it'

# For Gemini (cloud)
# MODEL_TYPE = 'gemini'
# MODEL_ID = 'gemini-2.5-flash'
# GEMINI_API_KEY = 'your-key-here'

# ===== GENERATION SETTINGS =====
GEMMA_CONFIG = {
    'max_context_length': 32000,  # Max input tokens
    'max_new_tokens': 256,        # Max output tokens
    'temperature': 0.1,           # Lower = more factual
    'device': 'cpu'               # or 'cuda' for GPU
}

GEMINI_CONFIG = {
    'max_output_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9
}
```

### Advanced Configuration

**Retrieval Settings** (in your code):
```python
from src.retrieval.pipeline import RetrievalConfig

config = RetrievalConfig(
    initial_k=20,        # Retrieve top 20 initially
    final_k=5,           # Return top 5 after re-ranking
    score_threshold=0.3, # Minimum relevance score
    use_sparse=False,    # Enable keyword search (BM25)
    dense_weight=0.7,    # Semantic search weight
    sparse_weight=0.3    # Keyword search weight
)
```

**Router Settings**:
```python
orchestrator = LangGraphOrchestrator(
    use_real_retrieval=True,      # Enable vector DB
    auto_load_docs=True,           # Auto-ingest documents
    use_keyword_routing=False,     # Use BERT (not keywords)
    use_gemma_generation=True      # Enable AI generation
)
```

## 📚 Usage Examples

### Example 1: Basic Query

```python
from src.router.orchestrator import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator(use_real_retrieval=True)
result = orchestrator.route_query("What is your return policy?")

print(result['answer'])
# Output: "Brooks offers a 90-day trial period. You can return 
#          shoes worn outdoors for a full refund within 90 days..."
```

### Example 2: Batch Processing

```python
queries = [
    "What is your return policy?",
    "How long do shoes last?",
    "What are shipping costs?"
]

for query in queries:
    result = orchestrator.route_query(query)
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {len(result.get('sources', []))}")
```

### Example 3: With Custom Configuration

```python
from src.generation.model_factory import ModelFactory

# Create custom generator
generator = ModelFactory.create_generator(
    model_type='gemini',
    model_config={
        'model_id': 'gemini-2.5-flash',
        'temperature': 0.3,
        'max_output_tokens': 256
    }
)

# Use in retrieval agent
from src.retrieval.agent import RetrievalAgent
agent = RetrievalAgent(
    use_vector_db=True,
    generator=generator
)

result = agent.handle_query(
    query="What is your return policy?",
    routing="retrieval",
    reasoning="User asking about policies"
)
```

### Example 4: Ingest New Documents

```python
from src.ingestion.pipeline import IngestionPipeline
from src.vectordb.store import VectorStore, HybridVectorStore

# Setup storage
vector_store = VectorStore()
hybrid_store = HybridVectorStore(vector_store)

# Create pipeline
pipeline = IngestionPipeline(
    hybrid_store,
    chunking_strategy="semantic",  # or "sentence", "fixed"
    chunk_size=500,
    overlap=100
)

# Ingest directory
record = pipeline.ingest_directory("./data/documents")
print(f"Ingested {record.total_chunks} chunks from {record.processed_documents} docs")

# Ingest single file
record = pipeline.ingest_file("./data/new_policy.pdf")
print(f"Added {record.total_chunks} chunks")
```

## 🧪 Testing

### Test Everything
```bash
pytest tests/ -v
```

### Test Specific Components
```bash
# Router
pytest tests/test_router.py -v

# Orchestrator
pytest tests/test_orchestrator.py -v

# Phase 2 (Vector DB + Retrieval)
pytest tests/test_phase2.py -v
```

### Test Model Performance
```bash
# Test Gemma generation speed
python test_gemma_speed.py

# Test embedding similarity
python test_embedding_semantic_match.py

# Test model factory
python test_model_factory.py
```

### Expected Test Results
```
tests/test_router.py ........................ PASSED
tests/test_orchestrator.py .................. PASSED  
tests/test_phase2.py ........................ PASSED

==================== 25 passed in 45.2s ====================
```

## 🎨 Using the Web Interface

### Starting the App
```bash
streamlit run app.py
```

### Features
- **Dark theme** - Easy on the eyes
- **Real-time responses** - See answers as they generate
- **Source citations** - Know where answers come from
- **Chat history** - Review previous questions
- **Model switching** - Change models in config

### Interface Guide

```
┌─────────────────────────────────────────────┐
│  Customer Support Assistant 🏃              │
├─────────────────────────────────────────────┤
│                                             │
│  💬 Ask me anything about:                │
│  • Return policies                          │
│  • Shipping information                     │
│  • Product details                          │
│  • Shoe recommendations                     │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │ Your question here...               │  │
│  └─────────────────────────────────────┘  │
│                           [Send] 📤       │
│                                             │
├─────────────────────────────────────────────┤
│  🤖 Assistant:                             │
│  Brooks offers a 90-day trial period...    │
│                                             │
│  📚 Sources:                               │
│  [1] return_policy.txt • Relevance: 0.95   │
│  [2] shipping_info.txt • Relevance: 0.87   │
└─────────────────────────────────────────────┘
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: "ChromaDB connection failed"

**Solution**:
```bash
# Delete and rebuild database
rm -rf chroma_db/
python -c "from src.vectordb.store import VectorStore; VectorStore()"
```

### Issue: "Gemma is very slow"

**Solutions**:
1. **Use GPU**:
   ```python
   # In src/config/model_config.py
   GEMMA_CONFIG = {'device': 'cuda'}
   ```

2. **Switch to Gemini**:
   ```bash
   export MODEL_TYPE='gemini'
   export GEMINI_API_KEY='your-key'
   ```

3. **Reduce token limits**:
   ```python
   GEMMA_CONFIG = {
       'max_context_length': 16000,  # Reduced
       'max_new_tokens': 128          # Reduced
   }
   ```

### Issue: "Out of memory"

**Solution**:
```python
# Use smaller context window
GEMMA_CONFIG = {
    'max_context_length': 8000,
    'max_new_tokens': 128
}

# Or use Gemini (no memory constraints)
MODEL_TYPE = 'gemini'
```

### Issue: "API key error" (Gemini)

**Solution**:
```bash
# Check key is set
echo $GEMINI_API_KEY

# Set key
export GEMINI_API_KEY='your-actual-key'

# Or hardcode in config (not recommended for production)
# src/config/model_config.py:
GEMINI_API_KEY = 'your-key-here'
```

## 🎯 Common Workflows

### Workflow 1: Development Testing

```bash
# 1. Use fast Gemini for development
export MODEL_TYPE='gemini'
export GEMINI_API_KEY='your-key'

# 2. Run app
streamlit run app.py

# 3. Test queries in browser
# 4. Check terminal for timing metrics
```

### Workflow 2: Production Deployment

```bash
# 1. Configure for production
# Edit src/config/model_config.py:
# MODEL_TYPE = 'gemini'
# MODEL_ID = 'gemini-2.5-flash'

# 2. Set environment variables
export GEMINI_API_KEY='prod-key'
export PYTHONUNBUFFERED=1

# 3. Run with proper logging
streamlit run app.py --server.port 8501 --server.headless true
```

### Workflow 3: Adding New Documents

```bash
# 1. Add documents to data/documents/
cp new_policy.pdf data/documents/

# 2. Restart app (auto-ingests)
streamlit run app.py

# 3. Or manually ingest
python -c "
from src.ingestion.pipeline import IngestionPipeline
from src.vectordb.store import VectorStore, HybridVectorStore

vs = VectorStore()
hs = HybridVectorStore(vs)
pipeline = IngestionPipeline(hs)
pipeline.ingest_file('data/documents/new_policy.pdf')
"
```

## 📊 Performance Benchmarks

### Router Performance
- Accuracy: **85.7%**
- Latency: **<500ms**
- Memory: **~1GB**

### Retrieval Performance
- Initial query: **3-5 seconds**
- Cached query: **<1 second**
- Precision@5: **~80%**

### Generation Performance
- **Gemma CPU**: 10-20 tok/s
- **Gemma GPU**: 40-60 tok/s
- **Gemini Flash**: 100+ tok/s
- **Gemini Pro**: 80+ tok/s

## 🚀 Next Steps

### Learn More
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Explore `examples/` folder for code samples
- Check `tests/` for usage patterns

### Customize
- Add your own documents to `data/documents/`
- Fine-tune the BERT router with `train_router.py`
- Adjust retrieval parameters for your use case
- Customize the Streamlit UI in `app.py`

### Deploy
- Set up environment variables
- Configure production database
- Enable logging and monitoring
- Deploy with Docker (coming soon)

## 📖 Additional Resources

- **Model Documentation**:
  - [Gemma 3](https://huggingface.co/google/gemma-3-270m-it)
  - [Gemini](https://ai.google.dev/tutorials/python_quickstart)
  
- **Framework Documentation**:
  - [LangChain](https://python.langchain.com/docs/get_started/introduction)
  - [ChromaDB](https://docs.trychroma.com/)
  - [Streamlit](https://docs.streamlit.io/)

## 💬 Getting Help

- Check troubleshooting section above
- Review test files for examples
- Open an issue on GitHub
- Read full documentation in README.md

---

**Ready to build? Start with the web interface: `streamlit run app.py`**
