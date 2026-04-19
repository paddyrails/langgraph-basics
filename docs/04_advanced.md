# LangGraph - Advanced

## Multi-Agent Patterns

### Supervisor Pattern
Central orchestrator routes tasks to specialist agents:
```
User --> Supervisor --> Billing Agent
                   +--> Support Agent
                   +--> Care Agent
```
Best for: clear domain boundaries, centralized control.

### Swarm Pattern
Agents hand off directly to each other (no central bottleneck):
```
Billing Agent <--> Support Agent <--> Care Agent
```
Best for: fluid conversations, peer-to-peer handoffs.

### Subgraph Agents
Each agent is its own compiled graph:
```python
billing_graph = StateGraph(BillingState)
# ... build full graph
billing_compiled = billing_graph.compile()

parent = StateGraph(ParentState)
parent.add_node("billing", billing_compiled)
```
Best for: complex agents with internal logic, team-owned graphs.

## Fan-Out / Fan-In (Map-Reduce)

```python
from langgraph.constants import Send

def scatter(state: State):
    return [Send("process_item", {"item": i}) for i in state["items"]]

graph.add_conditional_edges("start", scatter)
```

Parallel execution with results collected via reducer.

## Dynamic Graphs

Build graphs from config at runtime:

```python
def build_graph(config: dict) -> CompiledGraph:
    graph = StateGraph(State)
    for node_name, func in config["nodes"].items():
        graph.add_node(node_name, func)
    for src, dst in config["edges"]:
        graph.add_edge(src, dst)
    return graph.compile()
```

## Retry Policies

```python
from langgraph.pregel import RetryPolicy

policy = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)
graph.add_node("llm_call", agent_node, retry=policy)
```

## Message Trimming

Control context window growth:
```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(max_tokens=200, strategy="last", token_counter=llm)
trimmed = trimmer.invoke(state["messages"])
```

## Custom Reducers

```python
def dedup_reducer(existing: list, new: list) -> list:
    seen = {item["id"] for item in existing}
    return existing + [item for item in new if item["id"] not in seen]

class State(TypedDict):
    items: Annotated[list, dedup_reducer]
```
