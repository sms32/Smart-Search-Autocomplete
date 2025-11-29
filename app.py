from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
from typing import List
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI(title="Search Autocomplete v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "data/searches.jsonl"
INDEX_FILE = "index.faiss"

os.makedirs("data", exist_ok=True)

# Global model and index (loaded once)
model = None
index = None
dimension = 384

def normalize_embeddings(emb):
    """Normalize embeddings for cosine similarity"""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norm + 1e-8)  # Avoid division by zero

def load_model_and_index():
    """Load model and index on startup"""
    global model, index
    print("🚀 Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
    else:
        index = faiss.IndexFlatIP(dimension)
        print("📭 Created new FAISS index")
    
    return model, index

# Initialize on startup
model, index = load_model_and_index()

@app.on_event("startup")
async def rebuild_index():
    """Rebuild index from JSONL on startup (safely)"""
    global index
    if os.path.exists(DATA_FILE):
        searches = []
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("query") and len(entry["query"].strip()) >= 2:
                            searches.append(entry["query"])
                    except:
                        continue
            
            if searches:
                print(f"🔄 Rebuilding index from {len(searches)} searches...")
                embeddings = model.encode(searches, show_progress_bar=False)
                embeddings = normalize_embeddings(embeddings.astype('float32'))
                
                # Reset and rebuild
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                faiss.write_index(index, INDEX_FILE)
                print(f"✅ Rebuilt index with {index.ntotal} vectors")
        except Exception as e:
            print(f"⚠️ Index rebuild failed: {e}")

@app.post("/search")
async def log_search(request: dict):
    """Log search + update FAISS incrementally"""
    clean_query = request.get("query", "").strip()
    if len(clean_query) < 2:
        raise HTTPException(status_code=400, detail="Query too short (min 2 chars)")
    
    # Log to JSONL (append-only)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": clean_query
    }
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    
    # Incremental FAISS update
    embedding = model.encode([clean_query], show_progress_bar=False)
    embedding = normalize_embeddings(embedding.astype('float32'))
    index.add(embedding)
    faiss.write_index(index, INDEX_FILE)
    
    return {
        "status": "logged", 
        "query": clean_query,
        "total_searches": index.ntotal
    }

@app.get("/suggest")
async def suggest(query: str, k: int = 5):
    """Get top-k similar searches"""
    clean_query = query.strip()
    if len(clean_query) < 2:
        return []
    
    # Encode query
    query_embedding = model.encode([clean_query], show_progress_bar=False)
    query_embedding = normalize_embeddings(query_embedding.astype('float32'))
    
    # FAISS search (k+2 to handle edge cases)
    distances, indices = index.search(query_embedding, min(k + 2, index.ntotal))
    
    # Load searches in order
    suggestions = []
    search_count = 0
    
    if os.path.exists(DATA_FILE):
        all_searches = []
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("query"):
                            all_searches.append(entry)
                    except:
                        continue
        except:
            all_searches = []
        
        # Map indices to actual searches
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(all_searches) and search_count < k:
                # Skip duplicates and self-matches
                if i == 0 or all_searches[idx]["query"] != clean_query:
                    suggestions.append({
                        "query": all_searches[idx]["query"],
                        "score": max(0.0, float(dist)),  # Clamp negative scores
                        "rank": len(suggestions) + 1
                    })
                    search_count += 1
    
    return suggestions

@app.delete("/clear")
async def clear_data():
    """Reset all data"""
    global index
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    index = faiss.IndexFlatIP(dimension)
    faiss.write_index(index, INDEX_FILE)
    return {"status": "cleared", "total": 0}

@app.get("/recent")
async def get_recent(limit: int = 10):
    """Get recent searches for frontend display"""
    searches = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    entry = json.loads(line.strip())
                    searches.append({
                        "query": entry["query"],
                        "timestamp": entry["timestamp"]
                    })
                except:
                    continue
    return searches[::-1]  # Newest first

@app.get("/")
async def root():
    return {
        "message": "🚀 Search Autocomplete v3 - Ready!",
        "total_searches": index.ntotal,
        "model": "all-MiniLM-L6-v2",
        "endpoints": ["/suggest", "/search", "/recent", "/stats", "/clear"]
    }

@app.get("/stats")
async def stats():
    index_size = os.path.getsize(INDEX_FILE) / 1024 / 1024 if os.path.exists(INDEX_FILE) else 0
    jsonl_lines = len(open(DATA_FILE).readlines()) if os.path.exists(DATA_FILE) else 0
    return {
        "total_searches": index.ntotal,
        "jsonl_lines": jsonl_lines,
        "index_size_mb": round(index_size, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
