# src/rag.py
import os
import re
from typing import List, Tuple
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

# ---- Config ----
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")

# ---- Custom embedding function ----
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()

# ---- Global state ----
embed_fn = SentenceTransformerEmbeddingFunction(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=embed_fn
)

# ---- Helpers ----
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = re.split(r"\s+", text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_documents(files: List[str]):
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
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    scores = results["distances"][0]
    return list(zip(docs, scores))