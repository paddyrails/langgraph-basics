# LangGraph - Customizations

## Custom Reducers

Control how state merges happen:

```python
def priority_merge(existing: list, new: list) -> list:
    """Keep only high-priority items, dedup by ID."""
    combined = {item["id"]: item for item in existing + new}
    return sorted(combined.values(), key=lambda x: x["priority"], reverse=True)

class State(TypedDict):
    tasks: Annotated[list, priority_merge]
```

## Custom Nodes

Nodes are just functions - wrap any logic:

```python
async def cached_llm_node(state: State) -> dict:
    cache_key = hash(str(state["messages"][-1]))
    if cache_key in cache:
        return {"messages": [cache[cache_key]]}
    response = await llm.ainvoke(state["messages"])
    cache[cache_key] = response
    return {"messages": [response]}
```

## Custom Routing

Complex conditional logic beyond simple string matching:

```python
def smart_router(state: State) -> str:
    msg = state["messages"][-1].content.lower()
    if any(word in msg for word in ["bill", "invoice", "payment"]):
        return "billing"
    if state.get("escalated"):
        return "supervisor"
    if len(state["messages"]) > 20:
        return "summarize_and_end"
    return "general"
```

## Custom Checkpointers

Implement the `BaseCheckpointSaver` interface for any backend:

```python
class RedisCheckpointer(BaseCheckpointSaver):
    async def aget(self, config): ...
    async def aput(self, config, checkpoint, metadata): ...
    async def alist(self, config, *, limit=None): ...
```

## Custom Callback Handlers

Track metrics, costs, and events:

```python
class MetricsCollector(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output.get("token_usage", {})
        self.total_cost += self._calculate_cost(tokens)
```

## Graph Composition

Build reusable graph components:

```python
def create_agent_subgraph(name: str, tools: list, system_prompt: str):
    """Factory function for domain-specific agent subgraphs."""
    graph = StateGraph(AgentState)
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    # ... wire up nodes and edges
    return graph.compile()

billing_agent = create_agent_subgraph("billing", billing_tools, billing_prompt)
support_agent = create_agent_subgraph("support", support_tools, support_prompt)
```
