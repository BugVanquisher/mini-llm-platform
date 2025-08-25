# src/rag.py
import os
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# ---- Config ----
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss.index")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "data/docs.txt")

# ---- Global state ----
embedder = SentenceTransformer(EMBED_MODEL_NAME)
index = None
docs: List[str] = []


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = re.split(r"\s+", text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_documents(files: List[str]):
    """Load and index documents into FAISS."""
    global index, docs

    all_chunks = []
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    docs = all_chunks
    embeddings = embedder.encode(docs, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + docs
    Path("data").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOC_STORE_PATH, "w") as f:
        for d in docs:
            f.write(d + "\n\n")

    print(f"Ingested {len(docs)} chunks into FAISS.")


def load_index():
    """Load FAISS index + docs from disk."""
    global index, docs
    if Path(INDEX_PATH).exists() and Path(DOC_STORE_PATH).exists():
        index = faiss.read_index(INDEX_PATH)
        with open(DOC_STORE_PATH, "r") as f:
            docs = [x.strip() for x in f.read().split("\n\n") if x.strip()]
        print(f"Loaded {len(docs)} docs from FAISS.")
    else:
        raise RuntimeError("No index found. Run ingest_documents first.")


def query_index(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve top-k docs for a query."""
    if index is None or not docs:
        load_index()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    scores, idx = index.search(q_emb, top_k)
    results = [(docs[i], float(scores[0][j])) for j, i in enumerate(idx[0])]
    return results