"""                                                                           
LangGraph with LLMs — integration, streaming, Send API, subgraphs.            
Run: uv run python exercises/02_llm_and_streaming.py                          
"""                                                                           
                                                                            
import asyncio                                                                
import json                                                                   
from typing import Annotated, TypedDict                                       
                                                                            
from dotenv import load_dotenv                                              
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage,BaseMessage, SystemMessage                  
from langchain_openai import ChatOpenAI                                       
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages                              
from langgraph.checkpoint.memory import MemorySaver                         
from langgraph.types import Send                                              
                                                                            
load_dotenv()                                                                 

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)                          
                                                                            

# ============================================================                
# 1. LLM IN A NODE — basic chat
# ============================================================                
                                                                            
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]                      

                                                                            
async def chat_node(state: ChatState) -> dict:                              
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}                                           
                                            
                                                                            
async def demo_llm_node():                                                  
    print("\n=== 1. LLM IN A NODE ===")                                       
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)                                         
    graph.add_edge(START, "chat")                                           
    graph.add_edge("chat", END)         
    app = graph.compile()
                                                                            
    result = await app.ainvoke({"messages": [HumanMessage(content="What is LangGraph in one sentence?")]})                                               
    print(f"  {result['messages'][-1].content}")                            
                                                                            
                                        
# ============================================================                
# 2. STREAMING — token-by-token + node updates                                
# ============================================================
                                                                            
async def demo_streaming():                                                 
    print("\n=== 2. STREAMING (tokens + node updates) ===")                   
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)                                         
    graph.add_edge(START, "chat")                                           
    graph.add_edge("chat", END)
    app = graph.compile()                                                     

    print("  Tokens: ", end="", flush=True)                                   
    async for event, data in app.astream(                                   
        {"messages": [HumanMessage(content="Name 3 benefits of LangGraph. Be brief.")]},                                                                   
        stream_mode=["messages", "updates"],
    ):                                                                        
        if event == "messages":                                               
            message, metadata = data                                          
            if isinstance(message, AIMessageChunk) and message.content:       
                print(message.content, end="", flush=True)                  
        elif event == "updates":                                              
            pass  # node completed — could show progress here
                                                                            
    print()  # newline                                                        

                                                                            
# ============================================================              
# 3. MULTI-NODE WITH LLM — classify then respond                              
# ============================================================              
                                        
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]                      
    category: str
    answer: str                                                               
                                                                            

async def classifier(state: AgentState) -> dict:                              
    response = await llm.ainvoke([
        SystemMessage(content="Classify the query as 'technical' or 'general'.Reply with one word only."),                                               
        state["messages"][-1],          
    ])
    return {"category": response.content.strip().lower()}                     
                                            
                                                                            
async def technical_responder(state: AgentState) -> dict:                   
    response = await llm.ainvoke([                                            
        SystemMessage(content="You are a technical expert. Give a concise technical answer."),                                                          
        state["messages"][-1],                                              
    ])
    return {"answer": response.content, "messages":                           
[AIMessage(content=response.content)]}      
                                                                            
                                                                            
async def general_responder(state: AgentState) -> dict:                       
    response = await llm.ainvoke([
        SystemMessage(content="You are a friendly assistant. Give a simple    answer."),                                                                  
        state["messages"][-1],          
    ])
    return {"answer": response.content, "messages":                           
[AIMessage(content=response.content)]}      
                                                                            
                                                                            
def route_by_category(state: AgentState) -> str:                              
    return "technical" if "technical" in state["category"] else "general"
                                                                            
                                                                            
async def demo_multi_node():
    print("\n=== 3. MULTI-NODE (classify → route → respond) ===")             
    graph = StateGraph(AgentState)          
    graph.add_node("classify", classifier)                                    
    graph.add_node("technical", technical_responder)                        
    graph.add_node("general", general_responder)                              
                                            
    graph.add_edge(START, "classify")                                         
    graph.add_conditional_edges("classify", route_by_category, {            
        "technical": "technical",                                             
        "general": "general",               
    })                                                                        
    graph.add_edge("technical", END)                                        
    graph.add_edge("general", END)                                            

    app = graph.compile()                                                     
                                                                            
    for query in ["What is a mutex?", "What's a good movie to watch?"]:
        result = await app.ainvoke({    
            "messages": [HumanMessage(content=query)],
            "category": "",                                                   
            "answer": "",
        })                                                                    
        print(f"  '{query}' → [{result['category']}] {result['answer'][:80]}...")                
                                        

# ============================================================                
# 4. SEND API — fan-out / map-reduce
# ============================================================                
                                                                            
class DocState(TypedDict):
    doc: str                                                                  
    summary: str                            
                                                                            
                                                                            
class MapReduceState(TypedDict):                                              
    documents: list[str]
    summaries: Annotated[list[str], lambda old, new: old + new]               
    final_summary: str                                                      
                                        
                                            
async def summarize_doc(state: DocState) -> dict:                             
    response = await llm.ainvoke([
        SystemMessage(content="Summarize in one sentence."),                  
        HumanMessage(content=state["doc"]),                                 
    ])                                                                        
    return {"summary": response.content}                                    

                                                                            
async def fan_out(state: MapReduceState) -> list[Send]:
    """Send each document to its own summarize_doc node in parallel."""       
    return [Send("summarize", {"doc": doc, "summary": ""}) for doc in       
state["documents"]]                         
                                        
                                                                            
async def collect_summaries(state: MapReduceState) -> dict:                   
    """After all parallel summaries, reduce into one."""                      
    combined = "\n".join(state["summaries"])                                  
    response = await llm.ainvoke([                                          
        SystemMessage(content="Combine these summaries into one coherent paragraph."),                                                                 
        HumanMessage(content=combined), 
    ])                                                                        
    return {"final_summary": response.content}                                

                                                                            
# For Send API, each parallel node writes to parent state via reducer       
class SummarizeNodeState(TypedDict):
    doc: str                                                                  
    summary: str                        
                                                                            
                                                                            
async def summarize_node(state: SummarizeNodeState) -> dict:
    response = await llm.ainvoke([                                            
        SystemMessage(content="Summarize in one sentence."),                
        HumanMessage(content=state["doc"]),
    ])
    # This writes to parent's "summaries" list via Send                       
    return {"summaries": [response.content]}
                                                                            
                                                                            
async def demo_send_api():                                                    
    print("\n=== 4. SEND API (fan-out parallel summarization) ===")
                                                                            
    graph = StateGraph(MapReduceState)                                      
    graph.add_node("summarize", summarize_node)                               
    graph.add_node("reduce", collect_summaries)                             
                                        
    graph.add_conditional_edges(START, fan_out)
    graph.add_edge("summarize", "reduce")                                     
    graph.add_edge("reduce", END)
                                                                            
    app = graph.compile()                                                   
                                                                            
    docs = [                                                                
        "LangGraph is a framework for building stateful AI agents using graph-based workflows.",                    
        "FastAPI is a modern Python web framework for building APIs with      automatic OpenAPI docs.",
        "MongoDB Atlas provides managed database hosting with built-in vector search capabilities.",                                                      
    ]                                                                         
                                                                            
    result = await app.ainvoke({"documents": docs, "summaries": [], "final_summary": ""})
    print(f"  Individual summaries: {len(result['summaries'])}")
    print(f"  Combined: {result['final_summary'][:120]}...")                
                                        

# ============================================================                
# 5. SUBGRAPH — compiled graph as a node
# ============================================================                
                                                                            
class InnerState(TypedDict):
    query: str                                                                
    analysis: str
                                                                            
                                                                            
class OuterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]                      
    query: str
    analysis: str                                                             
    response: str                                                           
                                            
                                        
async def analyze(state: InnerState) -> dict:
    response = await llm.ainvoke([                                            
        SystemMessage(content="Analyze the sentiment and intent of this query.Be brief."),                                                                 
        HumanMessage(content=state["query"]),                               
    ])
    return {"analysis": response.content}                                     

                                                                            
def build_analysis_subgraph():                                              
    graph = StateGraph(InnerState)
    graph.add_node("analyze", analyze)                                        
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)                                            
    return graph.compile()                                                  
                                        

async def respond_with_analysis(state: OuterState) -> dict:                   
    response = await llm.ainvoke([
        SystemMessage(content=f"Analysis of user query: {state['analysis']}\nRespond helpfully based on this analysis."),
        state["messages"][-1],          
    ])
    return {"response": response.content, "messages":                         
[AIMessage(content=response.content)]}      
                                                                            
                                                                            
async def demo_subgraph():                                                    
    print("\n=== 5. SUBGRAPH (analysis subgraph inside parent) ===")
                                                                            
    inner = build_analysis_subgraph()                                       

    graph = StateGraph(OuterState)                                            
    graph.add_node("extract_query", lambda state: {"query": state["messages"][-1].content})                                               
    graph.add_node("analysis", inner)  # subgraph as node                   
    graph.add_node("respond", respond_with_analysis)
                                                                            
    graph.add_edge(START, "extract_query")
    graph.add_edge("extract_query", "analysis")                               
    graph.add_edge("analysis", "respond")                                     
    graph.add_edge("respond", END)                                            
                                                                            
    app = graph.compile()                                                   

    result = await app.ainvoke({            
        "messages": [HumanMessage(content="I'm frustrated that my order hasn't arrived yet!")],                           
        "query": "",                                                          
        "analysis": "",
        "response": "",                                                       
    })                                                                      
    print(f"  Analysis: {result['analysis'][:80]}...")                        
    print(f"  Response: {result['response'][:80]}...")                      

                                                                            
# ============================================================
# 6. MULTI-TURN WITH MEMORY — conversation continuity                         
# ============================================================              
                                        
async def demo_multi_turn():
    print("\n=== 6. MULTI-TURN CONVERSATION (checkpointer + thread_id) ===")  
    graph = StateGraph(ChatState)           
    graph.add_node("chat", chat_node)                                         
    graph.add_edge(START, "chat")                                           
    graph.add_edge("chat", END)                                               
    app = graph.compile(checkpointer=MemorySaver())
                                                                            
    config = {"configurable": {"thread_id": "convo-1"}}                       
                                        
    r1 = await app.ainvoke({"messages": [HumanMessage(content="My name is Paddy.")]}, config)                                                           
    print(f"  Turn 1: {r1['messages'][-1].content[:80]}...")
                                                                            
    r2 = await app.ainvoke({"messages": [HumanMessage(content="What's my name?")]}, config)                                                            
    print(f"  Turn 2: {r2['messages'][-1].content[:80]}...")                
    print(f"  Total messages in state: {len(r2['messages'])}")                
                                            
                                                                            
# ============================================================              
# RUN ALL                                                                     
# ============================================================
                                                                            
async def main():                                                           
    await demo_llm_node()
    await demo_streaming()                                                    
    await demo_multi_node()
    await demo_send_api()                                                     
    await demo_subgraph()                                                   
    await demo_multi_turn()
    print("\n✓ All LLM exercises complete") 
                                        
# uv run python exercises/02_llm_and_streaming.py
if __name__ == "__main__":
    asyncio.run(main())