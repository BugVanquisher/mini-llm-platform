# src/rag.py
import os
import re
from typing import List, Tuple
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---- Config ----
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")

# ---- Global state ----
embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_PATH)

def st_embeddings(texts: List[str]) -> List[List[float]]:
    """Custom embedding function for Chroma using SentenceTransformer."""
    return embedder.encode(texts).tolist()

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=st_embeddings
)

# ---- Helpers ----
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = re.split(r"\s+", text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_documents(files: List[str]):
    """Load and index documents into ChromaDB."""
    docs = []
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            chunks = chunk_text(text)
            docs.extend(chunks)

    ids = [f"doc_{i}" for i in range(len(docs))]
    collection.add(documents=docs, ids=ids)
    print(f"Ingested {len(docs)} chunks into ChromaDB.")

def query_index(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve top-k docs for a query from Chroma."""
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    scores = results["distances"][0]
    return list(zip(docs, scores))