# LangGraph - Deployment

## Deployment Options

### 1. LangGraph Platform (Managed)
- Hosted by LangChain Inc.
- Auto-scaling, managed infra
- Built-in persistence (PostgreSQL)
- Studio UI for debugging
- Best for: teams wanting zero-ops

### 2. Self-Hosted (Docker)
```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install langgraph langchain-openai
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0"]
```

### 3. Serverless (AWS Lambda / Cloud Functions)
- Wrap graph invocation in handler
- Use external checkpointer (DynamoDB/Firestore)
- Watch for cold-start latency

## Production Checkpointers

| Backend | Use Case |
|---------|----------|
| MemorySaver | Dev/testing only |
| PostgresSaver | Production default |
| MongoDBSaver | Document-heavy workloads |
| RedisSaver | High-throughput, ephemeral |

```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(conn_string="postgresql://...")
app = graph.compile(checkpointer=checkpointer)
```

## API Layer

Expose graph as REST API:
```python
from fastapi import FastAPI

app_api = FastAPI()

@app_api.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = await graph_app.ainvoke({"messages": request.messages}, config)
    return result
```

## Environment Configuration

- Store API keys in env vars / secret managers (never in code)
- Use `.env` files for local dev, cloud secret managers for prod
- Separate configs per environment (dev/staging/prod)

## CI/CD Considerations

- Test graphs with deterministic inputs (mock LLM responses)
- Validate graph structure: `graph.compile()` catches edge errors at build time
- Version checkpointer schemas alongside code
