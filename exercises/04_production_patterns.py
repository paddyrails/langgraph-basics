"""
Production patterns — retry_policy, parallel nodes, dynamic graphs, caching, token tracking.
Run: uv run python exercises/04_production_patterns.py
"""

import asyncio
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy, Send

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 1. RETRY POLICY — automatic retry on node failure
# ============================================================

call_count = {"value": 0}


async def flaky_node(state: dict) -> dict:
    """Fails first 2 times, succeeds on 3rd."""
    call_count["value"] += 1
    if call_count["value"] < 3:
        raise ConnectionError(f"Attempt {call_count['value']} failed")
    return {"result": f"Success on attempt {call_count['value']}"}


class RetryState(TypedDict):
    result: str


async def demo_retry_policy():
    print("\n=== 1. RETRY POLICY (auto-retry with backoff) ===")
    call_count["value"] = 0

    graph = StateGraph(RetryState)
    graph.add_node(
        "flaky",
        flaky_node,
        retry=RetryPolicy(
            max_attempts=5,
            initial_interval=0.1,  # seconds
            backoff_factor=2,
        ),
    )
    graph.add_edge(START, "flaky")
    graph.add_edge("flaky", END)
    app = graph.compile()

    result = await app.ainvoke({"result": ""})
    print(f"  {result['result']} (retried automatically)")


# ============================================================
# 2. PARALLEL NODES — multiple independent nodes at once
# ============================================================

class ParallelState(TypedDict):
    query: str
    weather: str
    news: str
    stock: str


async def fetch_weather(state: ParallelState) -> dict:
    await asyncio.sleep(0.5)  # simulate latency
    return {"weather": f"Weather for '{state['query']}': 28°C Sunny"}


async def fetch_news(state: ParallelState) -> dict:
    await asyncio.sleep(0.5)
    return {"news": f"News for '{state['query']}': Markets up 2%"}


async def fetch_stock(state: ParallelState) -> dict:
    await asyncio.sleep(0.5)
    return {"stock": f"Stock for '{state['query']}': $150.25"}


async def aggregate(state: ParallelState) -> dict:
    return {}  # all data already in state from parallel nodes


async def demo_parallel_nodes():
    print("\n=== 2. PARALLEL NODES (fan-out, fan-in) ===")

    graph = StateGraph(ParallelState)
    graph.add_node("weather", fetch_weather)
    graph.add_node("news", fetch_news)
    graph.add_node("stock", fetch_stock)
    graph.add_node("aggregate", aggregate)

    # Fan out: START → three nodes in parallel
    graph.add_edge(START, "weather")
    graph.add_edge(START, "news")
    graph.add_edge(START, "stock")

    # Fan in: all three → aggregate
    graph.add_edge("weather", "aggregate")
    graph.add_edge("news", "aggregate")
    graph.add_edge("stock", "aggregate")
    graph.add_edge("aggregate", END)

    app = graph.compile()

    start = time.time()
    result = await app.ainvoke({"query": "Mumbai", "weather": "", "news": "", "stock": ""})
    elapsed = time.time() - start

    print(f"  {result['weather']}")
    print(f"  {result['news']}")
    print(f"  {result['stock']}")
    print(f"  Time: {elapsed:.2f}s (parallel, not 1.5s sequential)")


# ============================================================
# 3. TOKEN TRACKING — monitor LLM usage per node
# ============================================================

class TrackedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    token_usage: dict


async def tracked_llm_node(state: TrackedState) -> dict:
    response = await llm.ainvoke(state["messages"])

    usage = {}
    if response.response_metadata:
        token_data = response.response_metadata.get("token_usage", {})
        usage = {
            "prompt_tokens": token_data.get("prompt_tokens", 0),
            "completion_tokens": token_data.get("completion_tokens", 0),
            "total_tokens": token_data.get("total_tokens", 0),
        }

    return {"messages": [response], "token_usage": usage}


async def demo_token_tracking():
    print("\n=== 3. TOKEN TRACKING (per-invocation usage) ===")

    graph = StateGraph(TrackedState)
    graph.add_node("chat", tracked_llm_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    result = await app.ainvoke({
        "messages": [HumanMessage(content="Explain LangGraph in 2 sentences.")],
        "token_usage": {},
    })

    usage = result["token_usage"]
    print(f"  Response: {result['messages'][-1].content[:80]}...")
    print(f"  Tokens — prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')}, total: {usage.get('total_tokens')}")


# ============================================================
# 4. DYNAMIC GRAPH — build graph from config at runtime
# ============================================================

class PipelineState(TypedDict):
    text: str
    steps_applied: Annotated[list[str], lambda old, new: old + new]


async def uppercase_step(state: PipelineState) -> dict:
    return {"text": state["text"].upper(), "steps_applied": ["uppercase"]}


async def strip_step(state: PipelineState) -> dict:
    return {"text": state["text"].strip(), "steps_applied": ["strip"]}


async def reverse_step(state: PipelineState) -> dict:
    return {"text": state["text"][::-1], "steps_applied": ["reverse"]}


async def prefix_step(state: PipelineState) -> dict:
    return {"text": f"[PROCESSED] {state['text']}", "steps_applied": ["prefix"]}


STEP_REGISTRY = {
    "uppercase": uppercase_step,
    "strip": strip_step,
    "reverse": reverse_step,
    "prefix": prefix_step,
}


def build_dynamic_pipeline(steps: list[str]):
    """Build a graph dynamically from a list of step names."""
    graph = StateGraph(PipelineState)

    for step_name in steps:
        graph.add_node(step_name, STEP_REGISTRY[step_name])

    # Chain them: START → step1 → step2 → ... → END
    graph.add_edge(START, steps[0])
    for i in range(len(steps) - 1):
        graph.add_edge(steps[i], steps[i + 1])
    graph.add_edge(steps[-1], END)

    return graph.compile()


async def demo_dynamic_graph():
    print("\n=== 4. DYNAMIC GRAPH (build from config) ===")

    # Config 1: strip → uppercase → prefix
    pipeline1 = build_dynamic_pipeline(["strip", "uppercase", "prefix"])
    r1 = await pipeline1.ainvoke({"text": "  hello world  ", "steps_applied": []})
    print(f"  Pipeline 1: '{r1['text']}' (steps: {r1['steps_applied']})")

    # Config 2: reverse → uppercase
    pipeline2 = build_dynamic_pipeline(["reverse", "uppercase"])
    r2 = await pipeline2.ainvoke({"text": "langgraph", "steps_applied": []})
    print(f"  Pipeline 2: '{r2['text']}' (steps: {r2['steps_applied']})")


# ============================================================
# 5. CACHING — avoid duplicate LLM calls
# ============================================================

class CachedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    cache_hit: bool


_llm_cache: dict[str, str] = {}


async def cached_llm_node(state: CachedState) -> dict:
    query = state["messages"][-1].content
    cache_key = query.lower().strip()

    if cache_key in _llm_cache:
        return {
            "messages": [AIMessage(content=_llm_cache[cache_key])],
            "cache_hit": True,
        }

    response = await llm.ainvoke(state["messages"])
    _llm_cache[cache_key] = response.content

    return {"messages": [response], "cache_hit": False}


async def demo_caching():
    print("\n=== 5. CACHING (skip duplicate LLM calls) ===")
    _llm_cache.clear()

    graph = StateGraph(CachedState)
    graph.add_node("chat", cached_llm_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    # First call — cache miss
    start = time.time()
    r1 = await app.ainvoke({"messages": [HumanMessage(content="What is FastAPI?")], "cache_hit": False})
    t1 = time.time() - start
    print(f"  Call 1: cache_hit={r1['cache_hit']}, time={t1:.2f}s")

    # Second call — cache hit (same query)
    start = time.time()
    r2 = await app.ainvoke({"messages": [HumanMessage(content="What is FastAPI?")], "cache_hit": False})
    t2 = time.time() - start
    print(f"  Call 2: cache_hit={r2['cache_hit']}, time={t2:.2f}s (instant)")


# ============================================================
# 6. NODE TIMING — latency profiling
# ============================================================

class TimedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    timings: dict


async def timed_classify(state: TimedState) -> dict:
    start = time.time()
    response = await llm.ainvoke([
        SystemMessage(content="Classify as 'question' or 'statement'. One word only."),
        state["messages"][-1],
    ])
    elapsed = time.time() - start
    return {
        "messages": [AIMessage(content=response.content)],
        "timings": {**state.get("timings", {}), "classify": round(elapsed * 1000)},
    }


async def timed_respond(state: TimedState) -> dict:
    start = time.time()
    response = await llm.ainvoke([
        SystemMessage(content="Respond concisely."),
        state["messages"][0],  # original user message
    ])
    elapsed = time.time() - start
    return {
        "messages": [AIMessage(content=response.content)],
        "timings": {**state.get("timings", {}), "respond": round(elapsed * 1000)},
    }


async def demo_node_timing():
    print("\n=== 6. NODE TIMING (latency profiling) ===")

    graph = StateGraph(TimedState)
    graph.add_node("classify", timed_classify)
    graph.add_node("respond", timed_respond)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    app = graph.compile()

    result = await app.ainvoke({
        "messages": [HumanMessage(content="How does LangGraph handle state?")],
        "timings": {},
    })

    print(f"  Response: {result['messages'][-1].content[:80]}...")
    print(f"  Timings: classify={result['timings']['classify']}ms, respond={result['timings']['respond']}ms")
    total = sum(result["timings"].values())
    print(f"  Total LLM time: {total}ms")


# ============================================================
# RUN ALL
# ============================================================

async def main():
    await demo_retry_policy()
    await demo_parallel_nodes()
    await demo_token_tracking()
    await demo_dynamic_graph()
    await demo_caching()
    await demo_node_timing()
    print("\n✓ All production pattern exercises complete")


if __name__ == "__main__":
    asyncio.run(main())
