"""
Multi-agent patterns — supervisor, swarm, handoff.
Run: uv run python exercises/06_multi_agent.py
"""

import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 1. SUPERVISOR PATTERN — orchestrator delegates to specialists
# ============================================================

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    billing_result: str
    support_result: str
    care_result: str
    final_response: str


async def supervisor(state: SupervisorState) -> dict:
    """Decides which specialist agent to delegate to."""
    response = await llm.ainvoke([
        SystemMessage(content="""You are a supervisor that routes queries to specialist agents.
Based on the query, respond with EXACTLY one of: billing, support, care.
- billing: invoices, payments, subscriptions, charges
- support: tickets, issues, complaints, SLA
- care: patients, visits, nursing, therapy, appointments
Reply with one word only."""),
        state["messages"][-1],
    ])
    return {"next_agent": response.content.strip().lower()}


async def billing_agent(state: SupervisorState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a billing specialist. Answer billing questions concisely. Mention invoice numbers and amounts."),
        state["messages"][-1],
    ])
    return {"billing_result": response.content}


async def support_agent(state: SupervisorState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a support specialist. Answer support questions concisely. Mention ticket IDs and SLA timelines."),
        state["messages"][-1],
    ])
    return {"support_result": response.content}


async def care_agent(state: SupervisorState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a care operations specialist. Answer care questions concisely. Mention patient protocols and visit types."),
        state["messages"][-1],
    ])
    return {"care_result": response.content}


async def synthesizer(state: SupervisorState) -> dict:
    """Combine specialist results into final response."""
    result = (
        state.get("billing_result", "")
        or state.get("support_result", "")
        or state.get("care_result", "")
    )
    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
    }


def route_to_agent(state: SupervisorState) -> str:
    agent = state["next_agent"]
    if "billing" in agent:
        return "billing"
    if "support" in agent:
        return "support"
    return "care"


async def demo_supervisor():
    print("\n=== 1. SUPERVISOR PATTERN (route to specialist agents) ===")

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("billing", billing_agent)
    graph.add_node("support", support_agent)
    graph.add_node("care", care_agent)
    graph.add_node("synthesize", synthesizer)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent, {
        "billing": "billing",
        "support": "support",
        "care": "care",
    })
    graph.add_edge("billing", "synthesize")
    graph.add_edge("support", "synthesize")
    graph.add_edge("care", "synthesize")
    graph.add_edge("synthesize", END)

    app = graph.compile()

    queries = [
        "What is the status of invoice INV-456?",
        "My ticket T-789 has been open for 3 days, what's the SLA?",
        "What protocols should I follow for a skilled nursing visit?",
    ]

    for query in queries:
        result = await app.ainvoke({
            "messages": [HumanMessage(content=query)],
            "next_agent": "", "billing_result": "", "support_result": "",
            "care_result": "", "final_response": "",
        })
        print(f"  Q: {query}")
        print(f"  Routed to: {result['next_agent']}")
        print(f"  A: {result['final_response'][:80]}...")
        print()


# ============================================================
# 2. SUPERVISOR WITH RE-DELEGATION — multi-step orchestration
# ============================================================

class MultiStepState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    agent_results: Annotated[list[str], lambda old, new: old + new]
    steps_taken: int
    final_response: str


MAX_STEPS = 3


async def smart_supervisor(state: MultiStepState) -> dict:
    """Supervisor that can delegate multiple times or finish."""
    results_so_far = "\n".join(state.get("agent_results", []))
    context = f"\nResults collected so far:\n{results_so_far}" if results_so_far else ""

    response = await llm.ainvoke([
        SystemMessage(content=f"""You are a supervisor orchestrating specialist agents.
Given the user query and any results collected so far, decide what to do next.
{context}

Options:
- "billing" — delegate to billing agent
- "support" — delegate to support agent
- "care" — delegate to care agent
- "done" — all needed info is collected, ready to respond

Reply with one word only."""),
        state["messages"][0],  # original query
    ])

    return {
        "next_agent": response.content.strip().lower(),
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


async def multi_billing(state: MultiStepState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a billing specialist. Answer concisely."),
        state["messages"][0],
    ])
    return {"agent_results": [f"[BILLING] {response.content}"]}


async def multi_support(state: MultiStepState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a support specialist. Answer concisely."),
        state["messages"][0],
    ])
    return {"agent_results": [f"[SUPPORT] {response.content}"]}


async def multi_care(state: MultiStepState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="You are a care specialist. Answer concisely."),
        state["messages"][0],
    ])
    return {"agent_results": [f"[CARE] {response.content}"]}


async def final_synthesizer(state: MultiStepState) -> dict:
    all_results = "\n".join(state.get("agent_results", []))
    response = await llm.ainvoke([
        SystemMessage(content="Combine these specialist results into one coherent answer."),
        HumanMessage(content=f"Original question: {state['messages'][0].content}\n\nSpecialist results:\n{all_results}"),
    ])
    return {"final_response": response.content, "messages": [AIMessage(content=response.content)]}


def route_multi_step(state: MultiStepState) -> str:
    if state["steps_taken"] >= MAX_STEPS:
        return "done"
    agent = state["next_agent"]
    if "done" in agent:
        return "done"
    if "billing" in agent:
        return "billing"
    if "support" in agent:
        return "support"
    if "care" in agent:
        return "care"
    return "done"


async def demo_multi_step_supervisor():
    print("\n=== 2. MULTI-STEP SUPERVISOR (re-delegation loop) ===")

    graph = StateGraph(MultiStepState)
    graph.add_node("supervisor", smart_supervisor)
    graph.add_node("billing", multi_billing)
    graph.add_node("support", multi_support)
    graph.add_node("care", multi_care)
    graph.add_node("synthesize", final_synthesizer)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_multi_step, {
        "billing": "billing",
        "support": "support",
        "care": "care",
        "done": "synthesize",
    })
    # After each agent, go back to supervisor for next decision
    graph.add_edge("billing", "supervisor")
    graph.add_edge("support", "supervisor")
    graph.add_edge("care", "supervisor")
    graph.add_edge("synthesize", END)

    app = graph.compile()

    # Cross-domain query that might need multiple agents
    result = await app.ainvoke({
        "messages": [HumanMessage(content="Is the billing correct for patient visit V123 and is there an open support ticket about it?")],
        "next_agent": "", "agent_results": [], "steps_taken": 0, "final_response": "",
    })

    print(f"  Steps taken: {result['steps_taken']}")
    print(f"  Agent results collected: {len(result['agent_results'])}")
    for r in result["agent_results"]:
        print(f"    {r[:80]}...")
    print(f"  Final: {result['final_response'][:100]}...")


# ============================================================
# 3. SWARM PATTERN — agents hand off to each other
# ============================================================

class SwarmState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_count: int
    response: str


async def swarm_billing(state: SwarmState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="""You are a billing agent in a swarm. Answer billing questions.
If the query also involves support issues, respond with: HANDOFF: support
If the query also involves care operations, respond with: HANDOFF: care
Otherwise, just answer the question."""),
        state["messages"][-1],
    ])
    content = response.content

    # Check for handoff
    if "HANDOFF: support" in content:
        return {"current_agent": "support", "handoff_count": state["handoff_count"] + 1}
    if "HANDOFF: care" in content:
        return {"current_agent": "care", "handoff_count": state["handoff_count"] + 1}

    return {"response": content, "current_agent": "done"}


async def swarm_support(state: SwarmState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="""You are a support agent in a swarm. Answer support questions.
If the query also involves billing, respond with: HANDOFF: billing
If the query also involves care, respond with: HANDOFF: care
Otherwise, just answer the question."""),
        state["messages"][-1],
    ])
    content = response.content

    if "HANDOFF: billing" in content:
        return {"current_agent": "billing", "handoff_count": state["handoff_count"] + 1}
    if "HANDOFF: care" in content:
        return {"current_agent": "care", "handoff_count": state["handoff_count"] + 1}

    return {"response": content, "current_agent": "done"}


async def swarm_care(state: SwarmState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="""You are a care operations agent in a swarm. Answer care questions.
If the query also involves billing, respond with: HANDOFF: billing
If the query also involves support, respond with: HANDOFF: support
Otherwise, just answer the question."""),
        state["messages"][-1],
    ])
    content = response.content

    if "HANDOFF: billing" in content:
        return {"current_agent": "billing", "handoff_count": state["handoff_count"] + 1}
    if "HANDOFF: support" in content:
        return {"current_agent": "support", "handoff_count": state["handoff_count"] + 1}

    return {"response": content, "current_agent": "done"}


MAX_HANDOFFS = 3


def swarm_entry(state: SwarmState) -> dict:
    """Determine initial agent from query keywords."""
    text = state["messages"][-1].content.lower()
    if any(kw in text for kw in ("invoice", "billing", "payment", "charge")):
        return {"current_agent": "billing"}
    if any(kw in text for kw in ("ticket", "issue", "complaint", "sla")):
        return {"current_agent": "support"}
    return {"current_agent": "care"}


def route_swarm(state: SwarmState) -> str:
    if state["handoff_count"] >= MAX_HANDOFFS:
        return "done"
    return state["current_agent"]


async def demo_swarm():
    print("\n=== 3. SWARM PATTERN (agents hand off to each other) ===")

    graph = StateGraph(SwarmState)
    graph.add_node("entry", swarm_entry)
    graph.add_node("billing", swarm_billing)
    graph.add_node("support", swarm_support)
    graph.add_node("care", swarm_care)

    graph.add_edge(START, "entry")
    graph.add_conditional_edges("entry", route_swarm, {
        "billing": "billing",
        "support": "support",
        "care": "care",
        "done": END,
    })

    # Each agent can hand off to any other or finish
    for agent in ("billing", "support", "care"):
        graph.add_conditional_edges(agent, route_swarm, {
            "billing": "billing",
            "support": "support",
            "care": "care",
            "done": END,
        })

    app = graph.compile()

    queries = [
        "What is the total on invoice INV-100?",
        "I have a complaint about my last care visit billing",
    ]

    for query in queries:
        result = await app.ainvoke({
            "messages": [HumanMessage(content=query)],
            "current_agent": "", "handoff_count": 0, "response": "",
        })
        print(f"  Q: {query}")
        print(f"  Handoffs: {result['handoff_count']}")
        print(f"  Final agent: {result['current_agent']}")
        print(f"  A: {result['response'][:80]}...")
        print()


# ============================================================
# 4. SUBGRAPH AGENTS — each agent is its own compiled graph
# ============================================================

class AgentState(TypedDict):
    query: str
    result: str


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str
    result: str
    final_response: str


def build_specialist_agent(persona: str):
    """Factory: build a specialist agent subgraph."""

    async def process(state: AgentState) -> dict:
        response = await llm.ainvoke([
            SystemMessage(content=persona),
            HumanMessage(content=state["query"]),
        ])
        return {"result": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph.compile()


# Build specialist subgraphs
billing_subgraph = build_specialist_agent(
    "You are a billing expert. Answer questions about invoices, payments, and subscriptions. Be concise."
)
support_subgraph = build_specialist_agent(
    "You are a support expert. Answer questions about tickets, SLAs, and escalations. Be concise."
)
care_subgraph = build_specialist_agent(
    "You are a care operations expert. Answer questions about patient visits, nursing protocols, and therapy. Be concise."
)


async def route_node(state: OrchestratorState) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="Classify as billing, support, or care. One word only."),
        state["messages"][-1],
    ])
    return {"route": response.content.strip().lower(), "query": state["messages"][-1].content}


def pick_agent(state: OrchestratorState) -> str:
    r = state["route"]
    if "billing" in r:
        return "billing_agent"
    if "support" in r:
        return "support_agent"
    return "care_agent"


async def format_response(state: OrchestratorState) -> dict:
    return {
        "final_response": state["result"],
        "messages": [AIMessage(content=state["result"])],
    }


async def demo_subgraph_agents():
    print("\n=== 4. SUBGRAPH AGENTS (each agent is its own compiled graph) ===")

    graph = StateGraph(OrchestratorState)
    graph.add_node("router", route_node)
    graph.add_node("billing_agent", billing_subgraph)   # subgraph
    graph.add_node("support_agent", support_subgraph)   # subgraph
    graph.add_node("care_agent", care_subgraph)         # subgraph
    graph.add_node("format", format_response)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", pick_agent, {
        "billing_agent": "billing_agent",
        "support_agent": "support_agent",
        "care_agent": "care_agent",
    })
    graph.add_edge("billing_agent", "format")
    graph.add_edge("support_agent", "format")
    graph.add_edge("care_agent", "format")
    graph.add_edge("format", END)

    app = graph.compile()

    result = await app.ainvoke({
        "messages": [HumanMessage(content="What are the nursing protocols for a skilled nursing visit?")],
        "query": "", "route": "", "result": "", "final_response": "",
    })

    print(f"  Routed to: {result['route']}")
    print(f"  Response: {result['final_response'][:100]}...")
    print(f"  (Each agent is an independent compiled graph used as a node)")


# ============================================================
# 5. COMPARISON — when to use which pattern
# ============================================================

async def demo_comparison():
    print("\n=== 5. MULTI-AGENT PATTERN COMPARISON ===")
    print("""
  SUPERVISOR:
    ✓ Central control — predictable, easy to debug
    ✓ Supervisor decides routing — can use LLM or rules
    ✓ Clean fan-out/fan-in — results merge at synthesizer
    ✗ Single point of failure — supervisor bottleneck
    Best for: RiteCare (clear BU boundaries, auditable routing)

  MULTI-STEP SUPERVISOR:
    ✓ Re-delegation — supervisor can call multiple agents sequentially
    ✓ Handles cross-domain queries naturally
    ✗ More LLM calls — supervisor called after each agent
    ✗ Needs max-step guard to prevent infinite loops
    Best for: Complex queries spanning multiple domains

  SWARM:
    ✓ No central bottleneck — agents route to each other
    ✓ Flexible — any agent can hand off to any other
    ✗ Hard to debug — unpredictable routing path
    ✗ Needs handoff limit — can loop between agents
    Best for: Peer-to-peer collaboration, exploratory tasks

  SUBGRAPH AGENTS:
    ✓ Encapsulation — each agent has its own graph, state, logic
    ✓ Testable independently — compile and test in isolation
    ✓ Reusable — same subgraph in different parent graphs
    ✗ State mapping overhead between parent and child
    Best for: When agents have different internal complexity
""")


# ============================================================
# RUN ALL
# ============================================================

async def main():
    await demo_supervisor()
    await demo_multi_step_supervisor()
    await demo_swarm()
    await demo_subgraph_agents()
    await demo_comparison()
    print("\n✓ All multi-agent exercises complete")


if __name__ == "__main__":
    asyncio.run(main())
