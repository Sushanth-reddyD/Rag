# Complete System Architecture - Phase 1 + Phase 2

## Overview

This document describes the complete architecture integrating Phase 1 (Router) with Phase 2 (Vector DB & Retrieval).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: ROUTER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Router Node (Keyword-based / LLM-based)                 │  │
│  │  • Analyzes query intent                                 │  │
│  │  • Classifies into 4 categories                          │  │
│  │  • Returns structured decision                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Category?      │
                    └─────┬───────────┘
         ┌───────────────┼───────────────┬──────────────┐
         │               │               │              │
    Complaint       API Call        Retrieval    Conversational
         │               │               │              │
         ▼               ▼               ▼              ▼
    ┌────────┐      ┌────────┐    ┌─────────┐    ┌──────────┐
    │Placeholder│    │Placeholder│  │PHASE 2  │    │Placeholder│
    │ Agent  │      │ Agent  │    │ Agent   │    │ Agent    │
    └────────┘      └────────┘    └────┬────┘    └──────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 2: RETRIEVAL SYSTEM                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. QUERY PROCESSING                                     │  │
│  │     • Normalize (lowercase, cleanup)                     │  │
│  │     • Expand (generate variations)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. VECTOR SEARCH (Dense)                                │  │
│  │     • Compute query embedding                            │  │
│  │     • Search ChromaDB (HNSW)                             │  │
│  │     • Get top-k similar chunks                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. SPARSE SEARCH (Optional)                             │  │
│  │     • BM25 keyword matching                              │  │
│  │     • Combine with dense results                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4. SCORE FUSION                                          │  │
│  │     • Weighted combination                                │  │
│  │     • Dense: 70%, Sparse: 30%                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  5. RE-RANKING                                            │  │
│  │     • Keyword overlap boost                               │  │
│  │     • Improve precision                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  6. POST-FILTERING                                        │  │
│  │     • Score threshold                                     │  │
│  │     • Deduplication                                       │  │
│  │     • Top-k selection                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  7. METADATA ENRICHMENT                                   │  │
│  │     • Load full metadata                                  │  │
│  │     • Add citations                                       │  │
│  │     • Format with provenance                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE WITH CITATIONS                      │
│  • Retrieved text chunks                                        │
│  • Source document info                                         │
│  • Relevance scores                                             │
│  • Chunk IDs for tracking                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow - Ingestion

```
┌─────────────┐
│  Documents  │ (PDF, HTML, DOCX, TXT)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  1. PREPROCESSING                   │
│     • Parse format-specific         │
│     • Extract structure             │
│     • Compute checksum              │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  2. CLEANING                        │
│     • Remove whitespace             │
│     • Normalize unicode             │
│     • Filter content                │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  3. CHUNKING                        │
│     Strategy:                       │
│     • Fixed: Overlapping windows    │
│     • Sentence: Natural boundaries  │
│     • Semantic: Structure-aware     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  4. EMBEDDING                       │
│     • Sentence Transformers         │
│     • all-MiniLM-L6-v2 (CPU)       │
│     • 384-dimensional vectors       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  5. METADATA ENRICHMENT             │
│     • Document metadata             │
│     • Chunk metadata                │
│     • Relationships (prev/next)     │
│     • Timestamps                    │
└──────┬──────────────────────────────┘
       │
       ├────────────────────┬──────────────────┐
       │                    │                  │
       ▼                    ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ChromaDB    │  │  Metadata    │  │  Tracking    │
│  (Vectors)   │  │  Store (JSON)│  │  (Logs)      │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID STORAGE                             │
├──────────────────────────────┬──────────────────────────────────┤
│  VECTOR STORE (ChromaDB)     │  METADATA STORE (JSON)           │
│  ┌────────────────────────┐  │  ┌────────────────────────────┐ │
│  │ • Embeddings (384-dim) │  │  │ • Full chunk text          │ │
│  │ • Minimal metadata     │  │  │ • Document metadata        │ │
│  │ • HNSW index           │  │  │ • Chunk relationships      │ │
│  │ • Fast similarity      │  │  │ • Citation info            │ │
│  │ • Persistence          │  │  │ • Timestamps               │ │
│  └────────────────────────┘  │  └────────────────────────────┘ │
│                              │                                  │
│  Used for:                   │  Used for:                       │
│  • Query matching            │  • Full text display             │
│  • Semantic search           │  • Citation formatting           │
│  • Top-k retrieval           │  • Provenance tracking           │
│  • Filtering                 │  • Detailed analysis             │
└──────────────────────────────┴──────────────────────────────────┘
```

## Component Interaction

```
┌───────────────────────────────────────────────────────────────────┐
│  Router (Phase 1)                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  LangGraphOrchestrator(use_real_retrieval=True)             │ │
│  │    │                                                          │ │
│  │    ├─> router_node (classify query)                          │ │
│  │    │                                                          │ │
│  │    └─> retrieval_agent (Phase 2)                             │ │
│  │         │                                                     │ │
│  │         ├─> RetrievalAgent                                   │ │
│  │         │    ├─> VectorStore                                 │ │
│  │         │    │    └─> ChromaDB                               │ │
│  │         │    │                                                │ │
│  │         │    └─> RetrievalPipeline                           │ │
│  │         │         ├─> QueryProcessor                         │ │
│  │         │         ├─> Dense search                           │ │
│  │         │         ├─> Sparse search (optional)               │ │
│  │         │         ├─> Fusion                                 │ │
│  │         │         ├─> Re-ranking                             │ │
│  │         │         └─> Post-filter                            │ │
│  │         │                                                     │ │
│  │         └─> Format response with citations                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
src/
├── router/
│   ├── orchestrator.py ───────┐
│   ├── router_node.py         │
│   └── routing_logic.py       │
│                               │
├── retrieval/              ◄───┘ (imports)
│   ├── agent.py ──────────┐
│   └── pipeline.py        │
│                           │
├── vectordb/            ◄──┘
│   └── store.py ──────────┐
│                           │
└── ingestion/          ◄───┘
    ├── models.py
    ├── preprocessing.py
    ├── chunking.py
    └── pipeline.py
```

## Configuration Flow

```
User Configuration
       │
       ├─> Phase 1 Config (src/config/settings.py)
       │   • Router model
       │   • Routing categories
       │   • Keyword patterns
       │
       └─> Phase 2 Config
           │
           ├─> Ingestion Config (pipeline init)
           │   • Chunking strategy
           │   • Chunk size
           │   • Overlap size
           │
           ├─> Vector Store Config (store init)
           │   • Collection name
           │   • Embedding model
           │   • Persist directory
           │
           └─> Retrieval Config (RetrievalConfig)
               • Initial k
               • Final k
               • Dense/sparse weights
               • Score threshold
```

## Summary

### Phase 1 (Router)
- **Purpose**: Classify user queries into categories
- **Methods**: Keyword-based (fast) or LLM-based (smart)
- **Output**: Routing decision + reasoning + confidence
- **Categories**: Complaint, API Call, Retrieval, Conversational

### Phase 2 (Retrieval)
- **Purpose**: Retrieve relevant information from documents
- **Method**: Multi-stage pipeline with vector + metadata
- **Output**: Ranked results with citations
- **Features**: Dense + sparse search, re-ranking, provenance

### Integration
- Phase 1 routes queries to appropriate agents
- Retrieval queries use Phase 2 system
- Other queries use placeholder agents (for now)
- Seamless user experience with structured output

### Next Steps (Phase 3)
- Implement remaining agents (API Call, Complaint, Conversational)
- Add conversation memory
- Create web API
- Implement feedback loop
