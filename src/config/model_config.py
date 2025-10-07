"""
Configuration file for RAG system

This file contains all configuration settings including:
- Model selection (Gemma or Gemini)
- API keys
- Model parameters
- System settings
"""

# ============================================================================
# GENERATION MODEL CONFIGURATION
# ============================================================================

# Model Type: 'gemma' or 'gemini'
MODEL_TYPE = 'gemma'  # Changed to 'gemma' - works without API key

# Model ID (specific model to use)
# For Gemma:
#   - 'google/gemma-3-270m-it' (default, fastest)
#   - 'google/gemma-2b'
#   - 'google/gemma-7b'
# For Gemini:
#   - 'gemini-2.5-flash' (default, fast & cheap)
#   - 'gemini-2.5-pro' (most powerful)
#   - 'gemini-2.0-flash-exp'
#   - 'gemini-1.5-pro'
#   - 'gemini-1.5-flash'
MODEL_ID = None  # None = use default for model type

# ============================================================================
# API KEYS
# ============================================================================

# Gemini API Key (required only if MODEL_TYPE = 'gemini')
# Get your key from: https://ai.google.dev/
GEMINI_API_KEY = "AIzaSyDPo4tY_40s-EYKus7xfwuO8H3yXTKnUUc"  # Set to 'your-api-key-here' to use Gemini

# ============================================================================
# GEMMA CONFIGURATION (Local Inference)
# ============================================================================

GEMMA_CONFIG = {
    'device': None,              # 'cuda', 'cpu', or None (auto-detect)
    'max_context_length': 32000,  # Maximum context tokens
    'max_new_tokens': 256,       # Maximum output tokens
    'temperature': 0.1,          # Sampling temperature (0.0-1.0)
    'top_p': 0.9,               # Nucleus sampling (0.0-1.0)
    'do_sample': True           # Use sampling vs greedy decoding
}

# ============================================================================
# GEMINI CONFIGURATION (API-based)
# ============================================================================

GEMINI_CONFIG = {
    'max_output_tokens': 512,    # Maximum output tokens
    'temperature': 0.1,          # Sampling temperature (0.0-1.0)
    'top_p': 0.9                # Nucleus sampling (0.0-1.0)
}

# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================

ROUTER_CONFIG = {
    'model_path': './models/fine_tuned_router',
    'use_fine_tuned': True,
    'confidence_threshold': 0.5
}

# ============================================================================
# VECTOR DATABASE CONFIGURATION
# ============================================================================

VECTORDB_CONFIG = {
    'collection_name': 'product_docs',
    'persist_directory': './chroma_db',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

RETRIEVAL_CONFIG = {
    'initial_k': 20,             # Initial retrieval candidates
    'final_k': 5,                # Final results to return
    'use_dense': True,           # Use dense retrieval
    'use_sparse': False,         # Use sparse retrieval (BM25)
    'use_reranking': True,       # Use reranking
    'min_score_threshold': 0.3   # Minimum relevance score
}

# ============================================================================
# DOCUMENT LOADING CONFIGURATION
# ============================================================================

DOCUMENT_CONFIG = {
    'documents_directory': './data/documents',
    'metadata_directory': './data/metadata',
    'manifest_file': './data/metadata/ingestion_manifest.json',
    'auto_load_on_startup': True
}

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

SYSTEM_CONFIG = {
    'use_real_retrieval': True,      # Use vector DB for retrieval
    'use_generation': True,          # Use LLM for answer generation
    'auto_load_documents': True,     # Auto-load docs on startup
    'verbose_logging': False         # Enable detailed logging
}

# ============================================================================
# QUICK PRESETS
# ============================================================================

# Uncomment one of these presets for quick configuration:

# PRESET: Local & Free (Gemma)
# MODEL_TYPE = 'gemma'
# MODEL_ID = 'google/gemma-3-270m-it'
# GEMINI_API_KEY = None

# PRESET: Fast & Cheap (Gemini Flash)
# MODEL_TYPE = 'gemini'
# MODEL_ID = 'gemini-2.5-flash'
# GEMINI_API_KEY = 'your-api-key-here'

# PRESET: Best Quality (Gemini Pro)
# MODEL_TYPE = 'gemini'
# MODEL_ID = 'gemini-2.5-pro'
# GEMINI_API_KEY = 'your-api-key-here'

# ============================================================================
# NOTES
# ============================================================================

# To switch models:
# 1. Change MODEL_TYPE to 'gemma' or 'gemini'
# 2. If using Gemini, set GEMINI_API_KEY
# 3. Optionally set MODEL_ID for specific model variant
# 4. Restart your application

# To get Gemini API key:
# Visit: https://ai.google.dev/
# Click "Get API Key" and copy it here
