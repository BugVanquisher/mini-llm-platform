# Mini LLM Platform

A lightweight **LLM platform & infra demo** for learning and experimentation.  
This project provides a FastAPI service that wraps **LLM providers** (e.g., Ollama, OpenAI), integrates **monitoring with Prometheus/Grafana**, and supports **Retrieval-Augmented Generation (RAG)** with document ingestion and vector search (ChromaDB).

---

## ✨ Features

- **FastAPI service** with `/generate` endpoint for text generation  
- **Provider abstraction**: swap between Ollama (local), OpenAI (API), or others  
- **System prompt enforcement** for consistent responses  
- **Prometheus metrics**: latency, errors, token usage, cost estimates  
- **RAG pipeline**:  
  - `/rag/ingest`: upload and index documents (using ChromaDB + SentenceTransformers)  
  - `/rag/query`: retrieve relevant docs, feed them into the LLM, return grounded answers with sources  
- **Docker Compose setup** with Prometheus + Grafana dashboards  
- **Extensible**: add new models, agents, tools, or evals  

---

## 🏗 Project Structure
```
src/
├── app.py              # FastAPI app (endpoints, metrics, orchestration)
├── rag.py              # RAG pipeline (ingestion, retrieval)
├── agent.py            # Agent loop (tools + reasoning)
├── metrics.py          # Prometheus metrics definitions
├── inference.py        # Provider abstractions (Ollama, OpenAI, etc.)
├── evals_rag.py        # Simple regression tests for RAG
configs/
├── model_config.yaml   # Config for provider, model name, system prompts
docker-compose.yml      # Services: API, Prometheus, Grafana
requirements.txt        # Python dependencies
```
---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/mini-llm-platform.git
cd mini-llm-platform
```

### 2. Install dependencies

Using Conda or venv:
```
pip install -r requirements.txt
```

### 3. Start services with Docker Compose
```
docker-compose up --build
```
```
This will launch:
	•	mini-llm-app (FastAPI LLM service at http://localhost:8000)
	•	Prometheus (metrics at http://localhost:9090)
	•	Grafana (dashboards at http://localhost:3000, default creds admin/admin)
```
⸻

## 🔧 Usage

Health check
```
curl http://localhost:8000/health
```
Generate text
```
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a 3-sentence summary of RAG for LLMs."}'
```
Ingest documents into RAG
```
curl -X POST "http://localhost:8000/rag/ingest" \
  -F "files=@docs/llm_notes.txt"
```
Query with RAG
```
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval-Augmented Generation?", "top_k": 3}'
```
Ask the agent
```
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 25 * 4 + 10?"}'
```
⸻

## 📊 Monitoring
Prometheus metrics exposed at:
```
http://localhost:8000/metrics
```

Key metrics:

	•	llm_request_latency_seconds (per provider/model)
	•	llm_tokens_total
	•	llm_cost_usd_total
	•	rag_queries_total
	•	rag_query_latency_seconds
    •	agent_queries_total, agent_tool_invocations_total, agent_latency_seconds
	•	Grafana dashboards available at http://localhost:3000

⸻

## 🧪 Testing & Evals

Run RAG evals:
```
python src/evals_rag.py
```
This will:

	•	Run a few QA pairs against /rag/query
	•	Check grounding against retrieved docs
	•	Report pass/fail rates

Agent evals (coming soon):
```
python src/evals_agent.py
```

⸻

## 🔮 Roadmap
✅ Phase 1: Core Infra
	•	FastAPI service
	•	LLM providers
	•	Prometheus + Grafana

✅ Phase 2: RAG
	•	Document ingestion & retrieval
	•	RAG metrics + dashboards

✅ Phase 3: Agents
	•	Agent loop (ReAct-style)
	•	RAG + Calculator tools
	•	Agent observability

🔜 Phase 4: Multi-Agent Orchestration
	•	Multi-agent collaboration (planner + worker agents)
	•	Workflow orchestration (task decomposition, parallelization)
	•	External connectors (databases, APIs, knowledge graphs)
	•	Agent evaluation & safety guardrails
⸻

## 📜 License

MIT License. Feel free to use and modify for your own experiments.

---

⚡ Now you can save this as `README.md` at the root of your repo.  

Would you like me to also generate a **ready-to-import Grafana dashboard JSON** that’s prewired to your `llm_request_latency_seconds`, `rag_queries_total`, and other metrics so you don’t have to build dashboards manually?