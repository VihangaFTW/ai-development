import os
from typing import TypedDict, List, Union, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import json


def _save_conversation_to_file(messages: List[Union[HumanMessage, AIMessage]], filename: str = "conversation_history.json", *kwargs) -> None:
    n =  len(messages)
    
    json_data = []    
    for i in range(n):
        if isinstance(messages[i], HumanMessage):
            json_data.append({"type": "Human", "content": messages[i].content}) 
        elif isinstance(messages[i], AIMessage):
            json_data.append({"type": "AI", "content": messages[i].content})
            
    with open(filename, "w") as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)
        
    print(f'Conversation saved to {filename}')
    
    
def _load_conversation_from_file(filename: str = "conversation_history.json") -> List[Union[HumanMessage, AIMessage]]:
    
    if not os.path.exists(filename):
        print(f'No file found at {filename}')
        return []
    
    conversation_history: List[Union[HumanMessage, AIMessage]] = []
    
    try:
        with open(filename, "r") as file:
            conversation_data =  json.load(file)
        
        for json_msg in conversation_data:
            if json_msg["type"] == "Human":
                conversation_history.append(HumanMessage(content=json_msg["content"]))
            elif json_msg["type"] == "AI":
                conversation_history.append(AIMessage(content=json_msg["content"]))
                
        print(f'Successfully loaded {len(conversation_history)} messages from {filename}')
    
    except (json.JSONDecodeError, KeyError) as err:
        print(f'There was an error while loading the conversation file: {err}')
        
        
        
    return conversation_history


load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    

# change model to liking. Cheapest model chosen as default.
gpt = ChatOpenAI(model="gpt-4.1-nano")


def processer(state: AgentState) -> AgentState:
    response = gpt.invoke(state["messages"])
    # print AI response
    print(f'\nAI: {response.content}')
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

graph = StateGraph(AgentState)
graph.add_node("process", processer)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history:  List[Union[HumanMessage, AIMessage]] = _load_conversation_from_file("conversation_history.json")

user_input = input("\nYou: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    
    conversation_history = result["messages"]
    user_input = input("\nYou: ")
    

_save_conversation_to_file(conversation_history, "conversation_history.json")






