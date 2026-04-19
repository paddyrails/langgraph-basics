# LangGraph - Scalability

## Horizontal Scaling

```
Load Balancer
    |
    +-- Worker 1 (LangGraph instance)
    +-- Worker 2 (LangGraph instance)
    +-- Worker N (LangGraph instance)
    |
Shared Checkpointer (PostgreSQL / Redis)
```

**Key requirement**: all workers must share the same checkpointer backend. MemorySaver is per-process and won't work across instances.

## Async-First Design

LangGraph is async-native. Always use async nodes:

```python
# Good - non-blocking
async def my_node(state: State) -> dict:
    result = await llm.ainvoke(state["messages"])
    return {"messages": [result]}

# Bad - blocks the event loop
def my_node(state: State) -> dict:
    result = llm.invoke(state["messages"])  # blocking!
    return {"messages": [result]}
```

## Parallel Node Execution

Fan-out for independent tasks:

```python
# These nodes have no dependency on each other
graph.add_edge(START, "fetch_weather")
graph.add_edge(START, "fetch_news")
graph.add_edge(START, "fetch_stocks")

# All three run concurrently, results merge via reducers
graph.add_edge("fetch_weather", "aggregate")
graph.add_edge("fetch_news", "aggregate")
graph.add_edge("fetch_stocks", "aggregate")
```

## Scaling Strategies by Layer

| Layer | Strategy |
|-------|----------|
| **API** | Multiple uvicorn workers behind LB |
| **Graph execution** | asyncio concurrency within process |
| **LLM calls** | Connection pooling, request batching |
| **Checkpointer** | PostgreSQL with connection pooling (pgbouncer) |
| **State** | Trim messages, limit history depth |
| **Long tasks** | Queue-based workers (Redis/SQS) |

## Caching for Scale

```python
from functools import lru_cache

# Cache deterministic LLM calls
response_cache = {}

async def cached_classify(text: str) -> str:
    if text in response_cache:
        return response_cache[text]
    result = await llm.ainvoke(f"Classify: {text}")
    response_cache[text] = result.content
    return result.content
```

## Scaling Checklist

- [ ] External checkpointer (not MemorySaver)
- [ ] Async nodes throughout
- [ ] Connection pooling for LLM and DB clients
- [ ] Message trimming to bound state size
- [ ] Queue-based processing for high throughput
- [ ] Monitoring for token costs and latency
- [ ] Horizontal scaling behind load balancer
- [ ] Checkpoint cleanup (TTL-based expiry)
