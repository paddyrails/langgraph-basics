# LangGraph - Integration

## LLM Integration

LangGraph is LLM-agnostic but typically uses LangChain chat models:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def agent_node(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

## Tool Integration

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72F and sunny in {city}"

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather])
```

**Tool execution node:**
```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([get_weather])
```

## Prebuilt ReAct Agent

One-liner for standard tool-calling agents:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[get_weather], checkpointer=saver)
result = await agent.ainvoke({"messages": [("user", "Weather in NYC?")]})
```

## Streaming

```python
async for event in app.astream(input_state, config):
    for node, output in event.items():
        print(f"{node}: {output}")

# Token-level streaming
async for event in app.astream_events(input_state, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

## Subgraphs (Graph Composition)

```python
# Build inner graph
inner = StateGraph(InnerState)
inner.add_node(...)
inner_compiled = inner.compile()

# Use as node in outer graph
outer = StateGraph(OuterState)
outer.add_node("analysis", inner_compiled)
```

## External Services

LangGraph nodes are just Python functions - integrate anything:
- **Databases**: MongoDB, PostgreSQL, Redis
- **APIs**: REST/gRPC calls inside nodes
- **Message Queues**: Kafka, SQS, RabbitMQ for async processing
- **Vector Stores**: FAISS, Pinecone, Chroma for RAG
