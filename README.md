# Smart Semantic Search Autocomplete

A production-ready semantic search autocomplete engine that learns from your search history using AI embeddings and FAISS vector search. Built with FastAPI backend and modern vanilla JS frontend. Understands context (e.g., "footb" → "messi vs ronaldo", "qiskit quantum").

## FEATURES
- Semantic autocomplete using all-MiniLM-L6-v2 embeddings (384-dim)
- FAISS vector search (<20ms even @ 10k searches)
- Real-time suggestions as you type (200ms debounce)
- Persistent JSONL storage + incremental FAISS updates
- Clickable recent searches history
- Production-ready: auto-index rebuild, CORS, error handling
- Zero external DB - single JSONL + FAISS file
- Semantic understanding: context-aware (football/ML/quantum)

## PROJECT STRUCTURE

search-autocomplete/
├── app.py                      # FastAPI backend (search, suggest, recent)
├── index.html                  # Vanilla JS frontend (search bar + history)
├── data/
│   └── searches.jsonl          # Append-only search history
├── index.faiss                 # FAISS vector index (~4MB @ 10k searches)
├── requirements.txt            # Python deps (FastAPI, sentence-transformers, faiss-cpu)
└── README.md                   # This file

