# LangGraph - Hacks & Alternates

## Useful Hacks

### 1. Skip Nodes Conditionally
```python
def maybe_skip(state: State) -> str:
    if state.get("skip_validation"):
        return "next_node"  # bypass validation
    return "validate"
```

### 2. Inject Debug State
```python
def debug_wrapper(func):
    async def wrapped(state):
        print(f"[DEBUG] Entering with keys: {list(state.keys())}")
        result = await func(state)
        print(f"[DEBUG] Returning: {list(result.keys())}")
        return result
    return wrapped

graph.add_node("my_node", debug_wrapper(my_func))
```

### 3. Early Exit Pattern
```python
def check_cache(state: State) -> str:
    if state["query"] in response_cache:
        return "return_cached"  # skip LLM entirely
    return "call_llm"
```

### 4. State as Feature Flag
Use state fields to toggle behavior without rebuilding graphs:
```python
class State(TypedDict):
    messages: list
    enable_rag: bool      # toggle RAG pipeline
    max_retries: int      # runtime-configurable
```

## Alternatives to LangGraph

| Framework | Strengths | When to Use |
|-----------|-----------|-------------|
| **CrewAI** | Role-based agents, simpler API | Multi-agent without graph complexity |
| **AutoGen** | Microsoft-backed, conversation patterns | Research, prototyping |
| **Temporal** | Durable execution, production-grade | Enterprise workflows needing ACID |
| **Prefect/Airflow** | Data pipeline orchestration | ETL, batch jobs (not real-time agents) |
| **Raw LangChain** | Simpler, less overhead | Linear chains, no cycles needed |
| **Haystack** | Pipeline-oriented, NLP-focused | Search/RAG-heavy applications |
| **DSPy** | Prompt optimization, declarative | When you want auto-tuned prompts |

## When to Hack vs Switch

**Stay with LangGraph** if:
- You need cycles, human-in-the-loop, or checkpointing
- Multi-agent with shared state
- Already invested in LangChain ecosystem

**Consider switching** if:
- You need distributed execution across machines (Temporal)
- Simple linear pipelines (raw LangChain)
- You want less abstraction (direct API calls)
