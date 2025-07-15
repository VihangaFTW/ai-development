from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    # Annotated[Sequence[BaseMessage], add_messages] adds metadata (add_messages) to the type hint, which LangGraph uses to specify how messages should be merged when updating the state
    #The add_messages function tells LangGraph to append new messages to existing ones rather than replacing them.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
    
@tool
def add(a: int, b: int) -> int:
    """
    This is an addition function that adds two numbers together.
    """
    
    return a + b


@tool
def character_count(string: str, character: str) -> int:
    """
    Count the number of occurrences of a specific character in a string.
    
    Args:
        string: The input string to search in (assumes lowercase ascii letters only).
        character: The character to count (assumes a single lowercase ascii letter).
        
    Returns:
        The number of times the character appears in the string.
    """
    count_array = [0] * (ord('z')-ord('a')+1)
    for s in string:
        count_index = ord(s)-ord('a')
        count_array[count_index] += 1
        
    
    return count_array[ord(character)-ord('a')]


tools  = [add, character_count]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    
    response =  model.invoke([system_prompt] + list(state["messages"]))
    
    return {
        "messages": [response]
    }
    
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    
    last_message = messages[-1]
    
    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    else:
        return "end"



graph  = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end": END,
        "continue": "tools"
    }
)

graph.add_edge("tools", "our_agent")
app = graph.compile()


def print_stream(stream) -> None:
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
            
            
inputs: AgentState = {"messages": [HumanMessage(content=" How many r's are in the word 'strawrrrberryabcgjsorngjrjgrjjfrhghv' ? ")]}
print_stream(app.stream(inputs, stream_mode="values"))