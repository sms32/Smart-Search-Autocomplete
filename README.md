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

```bash
## PROJECT STRUCTURE

search-autocomplete/
├── app.py                      # FastAPI backend (search, suggest, recent)
├── index.html                  # Vanilla JS frontend (search bar + history)
├── data/
│   └── searches.jsonl          # Append-only search history
├── index.faiss                 # FAISS vector index (~4MB @ 10k searches)
├── requirements.txt            # Python deps (FastAPI, sentence-transformers, faiss-cpu)
└── README.md                   # This file
```

## SETUP INSTRUCTIONS

 1. Clone the repository
```bash
git https://github.com/sms32/Smart-Search-Autocomplete.git
cd Smart-Search-Autocomplete
```
 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```
 3. Install dependencies
```bash
pip install -r requirements.txt
```
 4. Run the FastApi Server
```bash
uvicorn app:app --reload --port 8000
```
5. Frontend Setup
```bash
open live server in vs code for the index.html file
```

## LICENSE

This project is open-source and free to use for personal, academic, or non-commercial purposes.

