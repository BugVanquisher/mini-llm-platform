from prometheus_client import Counter, Histogram

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"]
)

# REQUEST_LATENCY = Histogram(
#     "llm_request_latency_seconds",
#     "Request latency in seconds",
#     ["endpoint", "method"]
# )

ERROR_COUNT = Counter(
    "llm_errors_total",
    "Total number of errors",
    ["endpoint", "method", "error_type"]
)

# Add token + cost metrics
TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed by the LLM",
    ["provider", "model"]
)

COST_TOTAL = Counter(
    "llm_cost_usd_total",
    "Estimated total cost (USD) of LLM calls",
    ["provider", "model"]
)

REQUEST_LATENCY_BY_MODEL = Histogram(
    "llm_request_latency_seconds",
    "Request latency in seconds, broken down by provider/model",
    ["provider", "model", "endpoint"]
)