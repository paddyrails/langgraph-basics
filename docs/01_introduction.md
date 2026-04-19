# LangGraph - Introduction

## What is LangGraph?

LangGraph is a **stateful orchestration framework** built on top of LangChain for building multi-step, cyclic AI agent workflows. Unlike simple chain-based pipelines, LangGraph models workflows as **directed graphs** with nodes (functions) and edges (transitions).

## Why LangGraph over plain LangChain?

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Flow control | Linear chains | Cycles, branches, fan-out |
| State | Passed through chain | Managed, typed, persistent |
| Human-in-the-loop | Manual | Built-in interrupt/resume |
| Multi-agent | Limited | Supervisor, Swarm, Subgraph |
| Checkpointing | No | Yes (time-travel, replay) |

## Core Mental Model

```
START --> Node A --> Conditional Edge --> Node B --> END
                        |
                        +--> Node C --> END
```

- **Nodes** = Python functions that transform state
- **Edges** = Transitions between nodes (static or conditional)
- **State** = TypedDict shared across all nodes
- **Reducers** = Rules for merging state updates (append vs replace)

## Minimal Example

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def greet(state: State) -> dict:
    return {"message": f"Hello, {state['message']}!"}

graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)
app = graph.compile()

result = app.invoke({"message": "World"})
# {'message': 'Hello, World!'}
```

## When to Use LangGraph

- Agent loops (ReAct, tool-calling)
- Multi-agent orchestration
- Workflows needing human approval
- Stateful conversations with memory
- Complex branching/retry logic
