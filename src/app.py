# src/app.py
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---- Config via env ----
PROVIDER = os.getenv("PROVIDER", "ollama")  # "ollama" | "openai"
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama3.1:8b" if PROVIDER == "ollama" else "gpt-4o-mini"
)

# Lazy import OpenAI only if used
openai_client = None
if PROVIDER == "openai":
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError("OpenAI provider selected but setup failed") from e

app = FastAPI(title="Mini LLM Platform", version="0.1.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    system: Optional[str] = None  # optional system prompt for OpenAI

@app.get("/health")
async def health():
    # Simple provider-specific health check
    if PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:11434/api/tags", timeout=3.0)
            ok = r.status_code == 200
        except Exception:
            ok = False
        return {"status": "ok" if ok else "down", "provider": PROVIDER, "model": MODEL_NAME}

    if PROVIDER == "openai":
        return {"status": "ok", "provider": PROVIDER, "model": MODEL_NAME}

    return {"status": "unknown provider", "provider": PROVIDER}

@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        if PROVIDER == "ollama":
            payload = {
                "model": MODEL_NAME,
                "prompt": req.prompt,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "num_predict": req.max_tokens
                }
            }
            async with httpx.AsyncClient() as client:
                r = await client.post("http://localhost:11434/api/generate", json=payload, timeout=60.0)
            r.raise_for_status()
            data = r.json()
            return {"provider": "ollama", "model": MODEL_NAME, "response": data.get("response", "")}

        elif PROVIDER == "openai":
            if openai_client is None:
                raise HTTPException(status_code=500, detail="OpenAI client not initialized")
            messages = []
            if req.system:
                messages.append({"role": "system", "content": req.system})
            messages.append({"role": "user", "content": req.prompt})
            chat = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
            )
            text = chat.choices[0].message.content
            return {"provider": "openai", "model": MODEL_NAME, "response": text}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {PROVIDER}")

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    # Using the in-process app reference avoids module import-path issues
    uvicorn.run(app, host="0.0.0.0", port=8000)