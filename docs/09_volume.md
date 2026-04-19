# LangGraph - Volume (High-Throughput Handling)

## Batch Processing

Process multiple inputs concurrently:

```python
import asyncio

async def process_batch(queries: list[str]):
    tasks = [
        graph_app.ainvoke(
            {"messages": [("user", q)]},
            {"configurable": {"thread_id": f"batch-{i}"}}
        )
        for i, q in enumerate(queries)
    ]
    return await asyncio.gather(*tasks)
```

## Connection Pooling

Reuse expensive clients across invocations:

```python
# Singleton pattern - create once, share everywhere
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", max_retries=2)
    return _llm
```

## Queue-Based Architecture

Decouple API from graph processing:

```
API --> Redis/SQS Queue --> Worker Pool --> Graph Execution --> Result Store
```

```python
# Worker pattern
async def worker(queue: asyncio.Queue):
    while True:
        request = await queue.get()
        result = await graph_app.ainvoke(request["input"], request["config"])
        await store_result(request["id"], result)
        queue.task_done()
```

## Message Trimming (Prevent Unbounded Growth)

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(max_tokens=500, strategy="last", token_counter=llm)
# Use in node before LLM call
state["messages"] = trimmer.invoke(state["messages"])
```

## Volume Considerations

| Scale | Approach |
|-------|----------|
| < 10 req/s | Direct `ainvoke`, no queue needed |
| 10-100 req/s | asyncio.gather + connection pooling |
| 100-1000 req/s | Queue-based workers (Redis/SQS) |
| 1000+ req/s | Horizontal scaling + load balancer + distributed checkpointer |

## Key Bottlenecks

1. **LLM API latency** (biggest) - mitigate with caching, parallel calls
2. **Checkpointer writes** - use async drivers, batch writes
3. **State serialization** - keep state lean, trim messages
4. **Memory** - large conversation histories eat RAM; use external storage
