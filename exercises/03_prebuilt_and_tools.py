"""
LangGraph prebuilt agents, tool calling, and ReAct pattern.
Run: uv run python exercises/03_prebuilt_and_tools.py
"""

import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 1. DEFINE TOOLS — LangChain @tool decorator
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake_data = {
        "mumbai": "32°C, Humid",
        "london": "15°C, Cloudy",
        "new york": "22°C, Sunny",
    }
    return fake_data.get(city.lower(), f"No weather data for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'"""
    try:
        return str(eval(expression))  # safe enough for demo
    except Exception as e:
        return f"Error: {e}"


@tool
def search_docs(query: str) -> str:
    """Search internal documents for information."""
    fake_docs = {
        "leave policy": "Employees get 24 days paid leave per year. Carry forward max 5 days.",
        "expense policy": "Expenses above $500 need manager approval. Submit within 30 days.",
        "remote work": "Hybrid model: 3 days office, 2 days remote. Full remote needs VP approval.",
    }
    for key, doc in fake_docs.items():
        if key in query.lower():
            return doc
    return "No relevant documents found."


tools = [get_weather, calculate, search_docs]


# ============================================================
# 2. PREBUILT REACT AGENT — one line
# ============================================================

async def demo_prebuilt_react():
    print("\n=== 1. PREBUILT REACT AGENT (create_react_agent) ===")

    agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "react-1"}}

    queries = [
        "What's the weather in Mumbai?",
        "Calculate 15 * 24 + 100",
        "What is our leave policy?",
    ]

    for query in queries:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config,
        )
        answer = result["messages"][-1].content
        print(f"  Q: {query}")
        print(f"  A: {answer[:100}}")
        print()


# ============================================================
# 3. PREBUILT WITH SYSTEM PROMPT — persona
# ============================================================

async def demo_react_with_prompt():
    print("\n=== 2. REACT WITH SYSTEM PROMPT ===")

    agent = create_react_agent(
        llm,
        tools,
        prompt="You are an HR assistant for RiteCare. Be concise and professional. Always cite the policy.",
    )

    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Can I work from home full time?")],
    })
    print(f"  {result['messages'][-1].content[:150}}...")


# ============================================================
# 4. CUSTOM REACT — build it yourself to understand internals
# ============================================================

class ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int


MAX_ITERATIONS = 5

# Bind tools to LLM — LLM will output tool_calls when needed
llm_with_tools = llm.bind_tools(tools)


async def agent_node(state: ReactState) -> dict:
    """Call LLM with tools. It decides whether to use a tool or respond."""
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}


async def tool_node(state: ReactState) -> dict:
    """Execute the tool calls from the LLM's response."""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        # Find and execute the tool
        tool_fn = {"get_weather": get_weather, "calculate": calculate, "search_docs": search_docs}[tool_call["name"]]
        result = await tool_fn.ainvoke(tool_call["args"])
        results.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call["id"],
        })

    return {"messages": results}


def should_continue(state: ReactState) -> str:
    """If LLM returned tool calls, execute them. Otherwise, done."""
    last = state["messages"][-1]

    # Safety: max iterations
    if state["iterations"] >= MAX_ITERATIONS:
        return "done"

    # If LLM wants to call tools
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "done"


async def demo_custom_react():
    print("\n=== 3. CUSTOM REACT AGENT (understand the internals) ===")

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "done": END,
    })
    graph.add_edge("tools", "agent")  # after tool execution, back to LLM

    app = graph.compile()

    # Multi-tool query — LLM will call multiple tools
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What's the weather in London and calculate 365 * 24")],
        "iterations": 0,
    })

    print(f"  Iterations: {result['iterations']}")
    print(f"  Answer: {result['messages'][-1].content[:150}}")

    # Show the full message chain
    print(f"\n  Message chain ({len(result['messages'])} messages):")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        content = str(msg.content)[:60] if msg.content else ""
        tool_calls = f" [calls: {[tc['name'] for tc in msg.tool_calls]}]" if hasattr(msg, "tool_calls") and msg.tool_calls else ""
        print(f"    {i}. {role}: {content}{tool_calls}")


# ============================================================
# 5. PREBUILT vs CUSTOM — side by side comparison
# ============================================================

async def demo_comparison():
    print("\n=== 4. PREBUILT vs CUSTOM COMPARISON ===")
    print("""
create_react_agent:
    ✓ One line to create
    ✓ Handles tool calling loop automatically
    ✓ Supports checkpointer, system prompt
    ✗ Can't customize the loop (no guardrails, no routing)
    ✗ Can't add nodes between agent and tools

Custom StateGraph:
    ✓ Full control — add guardrails, routing, retries
    ✓ Can add nodes anywhere in the flow
    ✓ Custom state fields beyond just messages
    ✗ More code to write
    ✗ Must handle tool execution yourself

When to use which:
    - Prototyping / simple agents → create_react_agent
    - Production / complex flows → Custom StateGraph (what RiteCare uses)
""")


# ============================================================
# 6. TOOL ERROR HANDLING — graceful failures
# ============================================================

@tool
def unreliable_api(query: str) -> str:
    """An API that sometimes fails."""
    raise ConnectionError("Service unavailable")


async def safe_tool_node(state: ReactState) -> dict:
    """Execute tools with error handling — write errors to state."""
    last_message = state["messages"][-1]
    results = []

    all_tools = {
        "get_weather": get_weather,
        "calculate": calculate,
        "search_docs": search_docs,
        "unreliable_api": unreliable_api,
    }

    for tool_call in last_message.tool_calls:
        tool_fn = all_tools.get(tool_call["name"])
        try:
            result = await tool_fn.ainvoke(tool_call["args"])
            content = str(result)
        except Exception as e:
            content = f"Tool error: {type(e).__name__}: {e}"

        results.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call["id"],
        })

    return {"messages": results}


async def demo_tool_error_handling():
    print("\n=== 5. TOOL ERROR HANDLING ===")

    llm_all_tools = llm.bind_tools(tools + [unreliable_api])

    async def agent_with_all_tools(state: ReactState) -> dict:
        response = await llm_all_tools.ainvoke(state["messages"])
        return {"messages": [response], "iterations": state["iterations"] + 1}

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_with_all_tools)
    graph.add_node("tools", safe_tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "done": END,
    })
    graph.add_edge("tools", "agent")
    app = graph.compile()

    result = await app.ainvoke({
        "messages": [HumanMessage(content="Call the unreliable API with query 'test'")],
        "iterations": 0,
    })
    print(f"  Answer: {result['messages'][-1].content[:150}}")
    print("  (LLM gracefully handled the tool error)")


# ============================================================
# RUN ALL
# ============================================================

async def main():
    await demo_prebuilt_react()
    await demo_react_with_prompt()
    await demo_custom_react()
    await demo_comparison()
    await demo_tool_error_handling()
    print("\n✓ All tool exercises complete")


if __name__ == "__main__":
    asyncio.run(main())
