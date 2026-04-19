# LangGraph - Observability

## Structured Logging

Add correlation IDs for request tracing:

```python
import logging, uuid

logger = logging.getLogger("langgraph")

async def node_with_logging(state: State) -> dict:
    request_id = state.get("request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Processing in node_a", extra={
        "request_id": request_id,
        "message_count": len(state["messages"]),
        "node": "node_a"
    })
    # ... node logic
    return result
```

## LLM Metrics Collection

Track token usage, cost, and latency:

```python
class MetricsCollector(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.calls.append({
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": self._calc_cost(usage),
            "timestamp": time.time()
        })
```

## Node-Level Metrics

Prometheus-style counters per node:

```python
node_metrics = {
    "calls": Counter("node_calls_total", "Total node calls", ["node"]),
    "errors": Counter("node_errors_total", "Total node errors", ["node"]),
    "latency": Histogram("node_latency_seconds", "Node latency", ["node"]),
}

def instrumented(node_name, func):
    async def wrapper(state):
        node_metrics["calls"].labels(node=node_name).inc()
        start = time.time()
        try:
            return await func(state)
        except Exception:
            node_metrics["errors"].labels(node=node_name).inc()
            raise
        finally:
            node_metrics["latency"].labels(node=node_name).observe(time.time() - start)
    return wrapper
```

## LangSmith Integration

```python
# Set env vars for automatic tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls_...
# LANGCHAIN_PROJECT=my-project

# Every graph invocation is automatically traced
# View at smith.langchain.com
```

## Key Metrics to Track

| Metric | Why |
|--------|-----|
| Tokens per request | Cost control |
| Node latency | Performance bottlenecks |
| Error rate per node | Reliability |
| Retry count | LLM flakiness |
| Messages per session | Context window pressure |
| Active threads | Concurrency load |

## Alerting Rules

- Token usage > 2x average (cost spike)
- Error rate > 5% on any node
- P99 latency > 30s
- Checkpoint storage > threshold (cleanup needed)
