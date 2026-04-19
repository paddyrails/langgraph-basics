# LangGraph - Foundation

## State Management

State is defined as a `TypedDict`. Every node receives the full state and returns a partial update.

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # append reducer
    count: int                                # replace reducer (default)
```

**Reducers** control how updates merge:
- **Replace** (default) - new value overwrites old
- **Append** (`add_messages`) - new values are appended to list
- **Custom** - write your own (e.g., deduplication)

## Graph Construction

```python
graph = StateGraph(State)
graph.add_node("node_a", func_a)
graph.add_node("node_b", func_b)

# Static edge
graph.add_edge(START, "node_a")

# Conditional edge
graph.add_conditional_edges("node_a", router_func, {"route1": "node_b", "route2": END})

graph.add_edge("node_b", END)
app = graph.compile()
```

## Conditional Edges

Router function returns a string key that maps to the next node:

```python
def router(state: State) -> str:
    if state["count"] > 3:
        return "done"
    return "retry"

graph.add_conditional_edges("process", router, {"done": END, "retry": "process"})
```

## Cycles

LangGraph supports loops - a node can route back to itself or an earlier node. This is how ReAct agents work (LLM -> Tool -> LLM -> ...).

## Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(state, config)
```

- Enables **conversation memory** across invocations
- Supports **time-travel** (replay from any checkpoint)
- Production: use PostgreSQL/MongoDB-backed checkpointers

## Human-in-the-Loop

```python
app = graph.compile(checkpointer=saver, interrupt_before=["sensitive_node"])
# Graph pauses before sensitive_node, user reviews, then:
app.invoke(None, config)  # resume
```
