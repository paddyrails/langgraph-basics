"""                                                                           
LangGraph Core Patterns — all key concepts in one file.                       
Run: uv run python exercises/01_core_patterns.py                            
"""                                     

import asyncio                                                                
from typing import Annotated, TypedDict 
                                                                            
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage      
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages                              
from langgraph.checkpoint.memory import MemorySaver                           
                                            
                                                                            
# ============================================================              
# 1. REDUCERS — how state merges work                                         
# ============================================================
                                                                            
class ChatState(TypedDict):                                                 
    messages: Annotated[list[BaseMessage], add_messages]  # append reducer    
    turn_count: int                                        # default: replace
                                                                            
                                                                            
def chat_node(state: ChatState) -> dict:                                      
    last = state["messages"][-1].content                                      
    return {                                                                  
        "messages": [AIMessage(content=f"Echo: {last}")],                     
        "turn_count": state["turn_count"] + 1,                              
    }                                                                         
                                            
                                                                            
def demo_reducer():                                                         
    print("\n=== 1. REDUCER (add_messages appends, turn_count replaces) ===") 
    graph = StateGraph(ChatState)           
    graph.add_node("chat", chat_node)                                         
    graph.add_edge(START, "chat")                                           
    graph.add_edge("chat", END)                                               
    app = graph.compile()
                                                                            
    result = app.invoke({                                                   
        "messages": [HumanMessage(content="Hello")],                          
        "turn_count": 0,                                                      
    })
    print(f"Messages: {len(result['messages'])} (Human + AI)")                
    print(f"Turn count: {result['turn_count']}")                            
                                            
                                        
# ============================================================                
# 2. CONDITIONAL EDGES — branching based on state                             
# ============================================================                
                                                                            
class RouterState(TypedDict):                                               
    query: str
    category: str                           
    answer: str                         

                                                                            
def classify(state: RouterState) -> dict:
    q = state["query"].lower()                                                
    if "price" in q or "cost" in q:                                         
        return {"category": "billing"}  
    elif "broken" in q or "fix" in q:
        return {"category": "support"}                                        
    return {"category": "general"}          
                                                                            
                                                                            
def billing_handler(state: RouterState) -> dict:                              
    return {"answer": "Routing to billing team..."}
                                                                            
                                                                            
def support_handler(state: RouterState) -> dict:
    return {"answer": "Creating support ticket..."}                           

                                                                            
def general_handler(state: RouterState) -> dict:                            
    return {"answer": "Here's general info..."}
                                            
                                        
def route_query(state: RouterState) -> str:
    return state["category"]                                                  
                                            
                                                                            
def demo_conditional_edges():                                               
    print("\n=== 2. CONDITIONAL EDGES (routing based on state) ===")          
    graph = StateGraph(RouterState)
    graph.add_node("classify", classify)                                      
    graph.add_node("billing", billing_handler)                              
    graph.add_node("support", support_handler)
    graph.add_node("general", general_handler)
                                                                            
    graph.add_edge(START, "classify")       
    graph.add_conditional_edges("classify", route_query, {                    
        "billing": "billing",                                               
        "support": "support",                                                 
        "general": "general",
    })                                                                        
    for node in ("billing", "support", "general"):                          
        graph.add_edge(node, END)
                                                                            
    app = graph.compile()
                                                                            
    for query in ["What's the price?", "My device is broken", "Tell me about services"]:                             
        result = app.invoke({"query": query, "category": "", "answer": ""})
        print(f"  '{query}' → {result['category']} → {result['answer']}")     
                                            
                                                                            
# ============================================================              
# 3. CYCLES — retry loop with counter                                         
# ============================================================
                                                                            
class RetryState(TypedDict):                                                
    value: int                                                                
    attempts: int
    done: bool                                                                
                                                                            
                                            
def process(state: RetryState) -> dict: 
    # Simulate: succeeds only when attempts >= 3
    attempts = state["attempts"] + 1                                          
    success = attempts >= 3             
    return {                                                                  
        "value": state["value"] * 2 if success else state["value"],           
        "attempts": attempts,               
        "done": success,                                                      
    }                                                                       
                                                                            

def should_retry(state: RetryState) -> str:                                   
    return "done" if state["done"] else "retry"                             
                                            
                                        
def demo_cycles():
    print("\n=== 3. CYCLES (retry loop with counter) ===")                    
    graph = StateGraph(RetryState)
    graph.add_node("process", process)                                        
    graph.add_edge(START, "process")                                        
    graph.add_conditional_edges("process", should_retry, {
        "retry": "process",             
        "done": END,
    })                                                                        
    app = graph.compile()
                                                                            
    result = app.invoke({"value": 10, "attempts": 0, "done": False})        
    print(f"  Final value: {result['value']}, Attempts: {result['attempts']}")
                                                                            
                                        
# ============================================================                
# 4. CHECKPOINTER — state persistence across invocations                      
# ============================================================
                                                                            
def demo_checkpointer():                                                    
    print("\n=== 4. CHECKPOINTER (MemorySaver, thread_id persistence) ===")   
    graph = StateGraph(ChatState)           
    graph.add_node("chat", chat_node)                                         
    graph.add_edge(START, "chat")                                           
    graph.add_edge("chat", END)                                               
    app = graph.compile(checkpointer=MemorySaver())
                                                                            
    config = {"configurable": {"thread_id": "session-1"}}                     
                                        
    # Turn 1                                                                  
    r1 = app.invoke({"messages": [HumanMessage(content="Hi")], "turn_count":  
0}, config)                                 
    print(f"  Turn 1: {len(r1['messages'])} messages")                        
                                                                            
    # Turn 2 — same thread_id, only send new message                          
    r2 = app.invoke({"messages": [HumanMessage(content="How are you?")],
"turn_count": 0}, config)                                                     
    print(f"  Turn 2: {len(r2['messages'])} messages (history preserved)")    
                                            
    # Different thread — isolated                                             
    config2 = {"configurable": {"thread_id": "session-2"}}                  
    r3 = app.invoke({"messages": [HumanMessage(content="New convo")],         
"turn_count": 0}, config2)                                                  
    print(f"  Session-2: {len(r3['messages'])} messages (isolated)")          
                                                                            
                                        
# ============================================================                
# 5. STATE HISTORY — time travel debugging                                    
# ============================================================
                                                                            
def demo_state_history():                                                   
    print("\n=== 5. STATE HISTORY (inspect every checkpoint) ===")            
    graph = StateGraph(RetryState)
    graph.add_node("process", process)                                        
    graph.add_edge(START, "process")                                        
    graph.add_conditional_edges("process", should_retry, {
        "retry": "process",                                                   
        "done": END,
    })                                                                        
    app = graph.compile(checkpointer=MemorySaver())                         

    config = {"configurable": {"thread_id": "retry-debug"}}                   
    app.invoke({"value": 10, "attempts": 0, "done": False}, config)
                                                                            
    for snapshot in app.get_state_history(config):                            
        step = snapshot.metadata.get("step", "?")
        vals = snapshot.values                                                
        print(f"  Step {step}: attempts={vals.get('attempts', '?')},          
done={vals.get('done', '?')}, value={vals.get('value', '?')}")
                                                                            
                                                                            
# ============================================================                
# 6. HUMAN-IN-THE-LOOP — interrupt, inspect, resume                           
# ============================================================                
                                                                            
class ApprovalState(TypedDict):
    request: str                                                              
    approved: bool
    result: str                                                               
                                                                            
                                        
def validate_request(state: ApprovalState) -> dict:
    return {"request": state["request"]}                                      

                                                                            
def execute_request(state: ApprovalState) -> dict:                          
    if state["approved"]:
        return {"result": f"Executed: {state['request']}"}                    
    return {"result": "Request was rejected by reviewer"}
                                                                            
                                                                            
def demo_human_in_the_loop():
    print("\n=== 6. HUMAN-IN-THE-LOOP (interrupt_before, resume) ===")        
    graph = StateGraph(ApprovalState)       
    graph.add_node("validate", validate_request)                              
    graph.add_node("execute", execute_request)                              
    graph.add_edge(START, "validate")                                         
    graph.add_edge("validate", "execute")   
    graph.add_edge("execute", END)                                            
                                                                            
    app = graph.compile(                                                      
        checkpointer=MemorySaver(),
        interrupt_before=["execute"],  # pause before execution               
    )                                                                       

    config = {"configurable": {"thread_id": "approval-1"}}                    
                                        
    # Step 1: invoke — will pause before "execute"                            
    app.invoke({"request": "Delete all records", "approved": False, "result": 
""}, config)
    paused_state = app.get_state(config)                                      
    print(f"  Paused before: {paused_state.next}")                          
    print(f"  Request: {paused_state.values['request']}")                     
                                                                            
    # Step 2: human reviews and approves                                      
    app.update_state(config, {"approved": True})                              
                                                                            
    # Step 3: resume                                                          
    result = app.invoke(None, config)                                       
    print(f"  Result: {result['result']}")                                    
                                                                            

# ============================================================                
# 7. CUSTOM REDUCER — deduplication
# ============================================================                
                                                                            
def dedup_list(existing: list[str], new: list[str]) -> list[str]:
    """Custom reducer: append only unique items."""                           
    seen = set(existing)                    
    return existing + [item for item in new if item not in seen]              
                                                                            
                                                                            
class TagState(TypedDict):
    text: str                                                                 
    tags: Annotated[list[str], dedup_list]                                  
                                                                            
                                                                            
def tag_extractor(state: TagState) -> dict:
    words = state["text"].lower().split()                                     
    return {"tags": [w for w in words if len(w) > 4]}
                                                                            
                                                                            
def demo_custom_reducer():
    print("\n=== 7. CUSTOM REDUCER (dedup_list) ===")                         
    graph = StateGraph(TagState)
    graph.add_node("extract", tag_extractor)                                  
    graph.add_edge(START, "extract")                                        
    graph.add_edge("extract", END)      
    app = graph.compile(checkpointer=MemorySaver())
                                                                            
    config = {"configurable": {"thread_id": "tags-1"}}
    r1 = app.invoke({"text": "LangGraph makes agent workflows simple", "tags":
[]}, config)                                                               
    print(f"  Turn 1 tags: {r1['tags']}")                                     

    r2 = app.invoke({"text": "LangGraph enables complex agent patterns",      
"tags": []}, config)                                                        
    print(f"  Turn 2 tags: {r2['tags']} (no duplicates)")                     
                                                                            
                                        
# ============================================================                
# RUN ALL                                                                     
# ============================================================
#cd langgraph-basics && uv sync && uv run python exercises/01_core_patterns.py                                                                            
if __name__ == "__main__":                                                  
    demo_reducer()
    demo_conditional_edges()
    demo_cycles()                           
    demo_checkpointer()                 
    demo_state_history()
    demo_human_in_the_loop()                                                  
    demo_custom_reducer()
    print("\n✓ All exercises complete")