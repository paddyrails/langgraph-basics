"""
Observability and scalability patterns — logging, callbacks, metrics, batching, trimming.
Run: uv run python exercises/07_observability_and_scale.py
"""

import asyncio
import time
import uuid
import logging
from typing import Annotated, TypedDict
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 1. STRUCTURED LOGGING — correlation IDs per request
# ============================================================

class LoggedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    response: str


# Simple structured logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("langgraph")


def log_node(node_name: str, request_id: str, **kwargs):
    """Structured log entry with correlation ID."""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"  [req={request_id}] node={node_name} {extra}")


async def logged_classify(state: LoggedState) -> dict:
    rid = state["request_id"]
    start = time.time()

    response = await llm.ainvoke([
        SystemMessage(content="Classify as 'question' or 'statement'. One word."),
        state["messages"][-1],
    ])

    log_node("classify", rid, result=response.content.strip(), latency_ms=round((time.time() - start) * 1000))
    return {"messages": [AIMessage(content=response.content)]}


async def logged_respond(state: LoggedState) -> dict:
    rid = state["request_id"]
    start = time.time()

    response = await llm.ainvoke([
        SystemMessage(content="Respond concisely."),
        state["messages"][0],
    ])

    log_node("respond", rid, tokens=len(response.content.split()), latency_ms=round((time.time() - start) * 1000))
    return {"response": response.content, "messages": [AIMessage(content=response.content)]}


async def demo_structured_logging():
    print("\n=== 1. STRUCTURED LOGGING (correlation IDs) ===")

    graph = StateGraph(LoggedState)
    graph.add_node("classify", logged_classify)
    graph.add_node("respond", logged_respond)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    app = graph.compile()

    request_id = str(uuid.uuid4())[:8]
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What are the care protocols for patient visits?")],
        "request_id": request_id,
        "response": "",
    })
    print(f"  Response: {result['response'][:60]}...")
    print(f"  (All logs share request_id={request_id} for tracing)")


# ============================================================
# 2. CUSTOM CALLBACK HANDLER — track every LLM call
# ============================================================

class MetricsCollector(BaseCallbackHandler):
    """Collects metrics from every LLM call in the graph."""

    def __init__(self):
        self.calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start = time.time()

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self._start
        self.total_latency += elapsed

        # Extract token usage from response
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            tokens = usage.get("total_tokens", 0)
            self.total_tokens += tokens
            # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
            prompt_cost = usage.get("prompt_tokens", 0) * 0.15 / 1_000_000
            completion_cost = usage.get("completion_tokens", 0) * 0.60 / 1_000_000
            self.total_cost += prompt_cost + completion_cost

        self.calls.append({
            "latency_ms": round(elapsed * 1000),
            "tokens": self.total_tokens,
        })

    def summary(self) -> str:
        return (
            f"LLM calls: {len(self.calls)}, "
            f"Total tokens: {self.total_tokens}, "
            f"Total latency: {round(self.total_latency * 1000)}ms, "
            f"Est. cost: ${self.total_cost:.6f}"
        )


class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    response: str


async def simple_chat(state: SimpleState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"response": response.content, "messages": [AIMessage(content=response.content)]}


async def demo_callback_metrics():
    print("\n=== 2. CUSTOM CALLBACK (metrics collection) ===")

    metrics = MetricsCollector()

    graph = StateGraph(SimpleState)
    graph.add_node("chat", simple_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    # Pass callback to track all LLM calls
    for query in ["What is LangGraph?", "Explain state management.", "What are reducers?"]:
        await app.ainvoke(
            {"messages": [HumanMessage(content=query)], "response": ""},
            {"callbacks": [metrics]},
        )

    print(f"  {metrics.summary()}")
    print(f"  Per-call breakdown:")
    for i, call in enumerate(metrics.calls):
        print(f"    Call {i+1}: {call['latency_ms']}ms")


# ============================================================
# 3. NODE-LEVEL METRICS — Prometheus-style counters
# ============================================================

class NodeMetrics:
    """Simple metrics registry — in production use prometheus_client."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.histograms: dict[str, list[float]] = {}

    def inc(self, name: str):
        self.counters[name] = self.counters.get(name, 0) + 1

    def observe(self, name: str, value: float):
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)

    def summary(self):
        lines = []
        for name, count in self.counters.items():
            lines.append(f"    {name}: {count}")
        for name, values in self.histograms.items():
            avg = sum(values) / len(values) if values else 0
            lines.append(f"    {name}: avg={avg:.0f}ms, count={len(values)}")
        return "\n".join(lines)


metrics_registry = NodeMetrics()


def tracked_node(node_name: str, fn):
    """Wrap a node function with metrics tracking."""
    async def wrapper(state):
        metrics_registry.inc(f"{node_name}_calls")
        start = time.time()
        try:
            result = await fn(state)
            metrics_registry.inc(f"{node_name}_success")
            return result
        except Exception as e:
            metrics_registry.inc(f"{node_name}_errors")
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            metrics_registry.observe(f"{node_name}_latency", elapsed)
    return wrapper


async def _classify(state: SimpleState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="Classify as question or statement. One word."),
        state["messages"][-1],
    ])
    return {"messages": [AIMessage(content=response.content)]}


async def _respond(state: SimpleState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="Respond concisely."),
        state["messages"][0],
    ])
    return {"response": response.content, "messages": [AIMessage(content=response.content)]}


async def demo_node_metrics():
    print("\n=== 3. NODE-LEVEL METRICS (Prometheus-style) ===")

    graph = StateGraph(SimpleState)
    graph.add_node("classify", tracked_node("classify", _classify))
    graph.add_node("respond", tracked_node("respond", _respond))
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    app = graph.compile()

    # Run 3 queries to build up metrics
    for query in [
        "What is LangGraph used for?",
        "Explain checkpointing in detail",
        "How do reducers work in state?",
    ]:
        await app.ainvoke({"messages": [HumanMessage(content=query)], "response": ""})

    print(f"  Metrics:\n{metrics_registry.summary()}")
    print("  (In production: expose via /metrics endpoint for Prometheus scraping)")


# ============================================================
# 4. MESSAGE TRIMMING — control context window growth
# ============================================================

class TrimmedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    response: str


async def trimmed_chat(state: TrimmedState) -> dict:
    # Trim to last N tokens before sending to LLM
    trimmed = trim_messages(
        state["messages"],
        max_tokens=200,
        strategy="last",
        token_counter=llm,
        start_on="human",       # always start from a human message
        allow_partial=False,
    )

    response = await llm.ainvoke(trimmed)
    return {"response": response.content, "messages": [AIMessage(content=response.content)]}


async def demo_message_trimming():
    print("\n=== 4. MESSAGE TRIMMING (context window control) ===")

    graph = StateGraph(TrimmedState)
    graph.add_node("chat", trimmed_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "trim-test"}}

    # Simulate a long conversation
    for i in range(10):
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=f"This is message number {i+1} about patient care protocols and visit scheduling details.")], "response": ""},
            config,
        )

    # Check how many messages are in state vs what LLM actually saw
    state = await app.aget_state(config)
    total_messages = len(state.values["messages"])
    print(f"  Total messages in state: {total_messages} (all 10 turns preserved)")
    print(f"  LLM only sees last ~200 tokens each time (trimmed before invocation)")
    print(f"  Last response: {result['response'][:60]}...")


# ============================================================
# 5. BATCH PROCESSING — multiple queries through the graph
# ============================================================

class BatchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    response: str


async def batch_chat(state: BatchState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"response": response.content, "messages": [AIMessage(content=response.content)]}


async def demo_batch_processing():
    print("\n=== 5. BATCH PROCESSING (concurrent graph invocations) ===")

    graph = StateGraph(BatchState)
    graph.add_node("chat", batch_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    queries = [
        "What is LangGraph?",
        "Explain FastAPI middleware.",
        "How does MongoDB vector search work?",
        "What is a Kafka consumer group?",
        "Describe the ReAct agent pattern.",
    ]

    # Sequential
    start = time.time()
    for q in queries:
        await app.ainvoke({"messages": [HumanMessage(content=q)], "response": ""})
    sequential_time = time.time() - start

    # Concurrent — all at once with asyncio.gather
    start = time.time()
    tasks = [
        app.ainvoke({"messages": [HumanMessage(content=q)], "response": ""})
        for q in queries
    ]
    results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start

    print(f"  Sequential ({len(queries)} queries): {sequential_time:.2f}s")
    print(f"  Concurrent ({len(queries)} queries): {concurrent_time:.2f}s")
    print(f"  Speedup: {sequential_time / concurrent_time:.1f}x")
    for r in results:
        print(f"    → {r['response'][:50]}...")


# ============================================================
# 6. CONNECTION POOLING — reuse LLM clients
# ============================================================

async def demo_connection_pooling():
    print("\n=== 6. CONNECTION POOLING (singleton pattern) ===")
    print("""
  WRONG — new client per request:
    async def my_node(state):
        llm = ChatOpenAI(model="gpt-4o-mini")  # new connection every call
        return await llm.ainvoke(...)

  RIGHT — module-level singleton:
    llm = ChatOpenAI(model="gpt-4o-mini")  # created once at import

    async def my_node(state):
        return await llm.ainvoke(...)  # reuses connection pool

  In RiteCare:
    - ChatOpenAI in responder.py: module-level ✓
    - ChatOpenAI in intent_classifier.py: module-level ✓
    - ChatOpenAI in guardrails.py: lazy singleton ✓
    - Motor (MongoDB): singleton via get_database() ✓
    - httpx: use AsyncClient with connection pooling for BU service calls

  For httpx (calling BU services from agent):
    # WRONG
    async with httpx.AsyncClient() as client:  # new pool per call
        resp = await client.get(url)

    # RIGHT — reuse across requests
    _client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=httpx.Timeout(30.0),
    )

    async def call_bu_service(url):
        return await _client.get(url)
""")


# ============================================================
# 7. QUEUE-BASED PROCESSING — decouple API from graph
# ============================================================

async def demo_queue_pattern():
    print("\n=== 7. QUEUE-BASED PROCESSING (decouple API from graph) ===")

    # Simulate an in-memory queue (production: Redis, Kafka, SQS)
    queue: asyncio.Queue = asyncio.Queue()
    results: dict[str, str] = {}

    graph = StateGraph(SimpleState)
    graph.add_node("chat", simple_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    async def worker():
        """Background worker that processes graph invocations from queue."""
        while True:
            job_id, query = await queue.get()
            try:
                result = await app.ainvoke({
                    "messages": [HumanMessage(content=query)],
                    "response": "",
                })
                results[job_id] = result["response"]
            except Exception as e:
                results[job_id] = f"Error: {e}"
            queue.task_done()

    # Start 2 workers
    workers = [asyncio.create_task(worker()) for _ in range(2)]

    # Enqueue jobs (simulates API receiving requests)
    jobs = {}
    for query in ["What is LangGraph?", "Explain checkpointing.", "What are subgraphs?"]:
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = query
        await queue.put((job_id, query))

    # Wait for all jobs to complete
    await queue.join()

    # Cancel workers
    for w in workers:
        w.cancel()

    print(f"  Processed {len(jobs)} jobs via queue:")
    for job_id, query in jobs.items():
        print(f"    [{job_id}] '{query[:30]}' → {results[job_id][:40]}...")

    print("""
  Production pattern:
    API endpoint → enqueue job → return job_id immediately
    Background workers → dequeue → graph.ainvoke() → store result
    Client polls GET /result/{job_id} or receives webhook

  Benefits:
    - API never blocks on LLM calls
    - Workers scale independently (add more pods)
    - Built-in backpressure (queue size limit)
    - Retry failed jobs from the queue
""")


# ============================================================
# RUN ALL
# ============================================================

async def main():
    await demo_structured_logging()
    await demo_callback_metrics()
    await demo_node_metrics()
    await demo_message_trimming()
    await demo_batch_processing()
    await demo_connection_pooling()
    await demo_queue_pattern()
    print("\n✓ All observability and scale exercises complete")


if __name__ == "__main__":
    asyncio.run(main())
