# src/app.py
import os
from typing import List, Optional
import time

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn
import httpx
from dotenv import load_dotenv
from .rag import ingest_documents, query_index
from .metrics import REQUEST_COUNT, ERROR_COUNT, TOKENS_TOTAL, COST_TOTAL, REQUEST_LATENCY_BY_MODEL, RAG_QUERIES_TOTAL, RAG_RETRIEVED_DOCS, RAG_LATENCY

load_dotenv()

# ---- Config via env ----
PROVIDER = os.getenv("PROVIDER", "ollama")  # "ollama" | "openai"
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b" if PROVIDER == "ollama" else "gpt-4o-mini")

# 🔑 Default system prompt (can override in request or via env)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are an expert ML systems engineer. "
    "Always interpret 'RAG' as Retrieval-Augmented Generation in the context of LLMs, "
    "not Red-Amber-Green project tracking. "
    "Keep answers concise, accurate, and technical."
)

import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response



OLLAMA_URL = "http://host.docker.internal:11434"

# Lazy import OpenAI client only if needed
openai_client = None
if PROVIDER == "openai":
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError("OpenAI provider selected but setup failed") from e

app = FastAPI(title="Mini LLM Platform", version="0.2.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    system: Optional[str] = None  # override system prompt

@app.get("/health")
async def health():
    if PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
            ok = r.status_code == 200
        except Exception:
            ok = False
        return {"status": "ok" if ok else "down", "provider": PROVIDER, "model": MODEL_NAME}

    if PROVIDER == "openai":
        return {"status": "ok", "provider": PROVIDER, "model": MODEL_NAME}

    return {"status": "unknown provider", "provider": PROVIDER}

# --- Expose metrics endpoint ---
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Resolve system prompt (env default → request override)
    system_prompt = req.system or DEFAULT_SYSTEM_PROMPT
    start_time = time.time()
    endpoint = "/generate"
    method = "POST"
    status = "200"

    try:
        if PROVIDER == "ollama":
            payload = {
                "model": MODEL_NAME,
                # Concatenate system + user prompt (Ollama doesn’t have chat roles)
                "prompt": f"{system_prompt}\n\nUser: {req.prompt}\n\nAnswer:",
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "num_predict": req.max_tokens
                }
            }
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60.0)
            r.raise_for_status()
            data = r.json()
            duration = time.time() - start_time
            REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
            # REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
            # Extract token usage
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            # For Ollama: no official $ pricing — assume $0.000001 per token (≈ GPT-3.5 pricing) for demo
            cost_estimate = total_tokens * 0.000001

            # Record metrics
            TOKENS_TOTAL.labels(provider="ollama", model=MODEL_NAME).inc(total_tokens)
            COST_TOTAL.labels(provider="ollama", model=MODEL_NAME).inc(cost_estimate)

            # After getting response
            duration = time.time() - start_time
            REQUEST_LATENCY_BY_MODEL.labels(
                provider="ollama",
                model=MODEL_NAME,
                endpoint=endpoint
            ).observe(duration)

            return {
                "provider": "ollama",
                "model": MODEL_NAME,
                "system_prompt": system_prompt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": round(duration, 3),
                "cost_estimate_usd": round(cost_estimate, 6),
                "response": data.get("response", "")
            }

        elif PROVIDER == "openai":
            if openai_client is None:
                raise HTTPException(status_code=500, detail="OpenAI client not initialized")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt},
            ]
            chat = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
            )
            text = chat.choices[0].message.content
            # duration = time.time() - start_time
            # REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
            # REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
            return {
                "provider": "openai",
                "model": MODEL_NAME,
                "system_prompt": system_prompt,
                "response": text
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {PROVIDER}")
    
    except httpx.HTTPError as e:
        status = "502"
        ERROR_COUNT.labels(endpoint=endpoint, method=method, error_type="http").inc()
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}") from e

    except Exception as e:
        status = "500"
        ERROR_COUNT.labels(endpoint=endpoint, method=method, error_type="internal").inc()
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    
@app.post("/rag/ingest")
async def rag_ingest(files: List[UploadFile]):
    paths = []
    for file in files:
        content = await file.read()
        path = f"data/{file.filename}"
        with open(path, "wb") as f:
            f.write(content)
        paths.append(path)

    ingest_documents(paths)
    return {"status": "ok", "ingested_files": paths}


class RagQuery(BaseModel):
    query: str
    top_k: int = 3

@app.post("/rag/query")
async def rag_query(req: RagQuery):
    start_time = time.time()
    results = query_index(req.query, req.top_k)

    # Prometheus metrics
    RAG_QUERIES_TOTAL.inc()
    RAG_RETRIEVED_DOCS.observe(len(results))
    RAG_LATENCY.observe(time.time() - start_time)

    # Build context for LLM
    context = "\n\n".join([r[0] for r in results])
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Context:\n{context}\n\nQuestion: {req.query}\nAnswer:",
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60.0)
    r.raise_for_status()
    answer = r.json().get("response", "")

    return {
        "query": req.query,
        "answer": answer,
        "sources": [r[0] for r in results]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)