from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from dotenv import load_dotenv
import json

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document: str
    

@tool
def update_doc(content: str) -> str:
    """
    Update the document content by replacing the current document.
    
    This tool replaces the entire document content with the provided content.
    It returns a JSON string containing the operation status and the new content.
    
    Args:
        content (str): The new content to replace the current document with.
        
    Returns:
        str: A JSON string containing:
            - status: "success" indicating the operation completed
            - message: Confirmation message about the update
            - content: The new document content that was set
    """
    
    result = {
        "status" : "success",
        "message" : "Document updated successfully",
        "content" : content
    }
    
    return json.dumps(result)

def process_doc_update(state: AgentState) -> AgentState:
    """
    Update the document content in the agent state based on the last tool message.
    
    This function processes the result from the update_doc tool call by extracting
    the new document content from the tool message and updating the agent state.
    If the tool message is malformed or missing, it falls back to the previous
    document content to maintain state consistency.
    
    Args:
        state (AgentState): The current agent state containing messages and document.
        
    Returns:
        AgentState: Updated state with the new document content, or unchanged state
                   if the tool message processing fails.
    """
    
    # get the last message which is a ToolMessage from update_doc tool call.
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage):
        try:
            tool_msg = json.loads(str(last_message.content))
            new_doc =  tool_msg.get("content", "")
        except json.JSONDecodeError:
            # fallback to the previous document content 
            new_doc  = state["document"]
            
        return {
            **state,
            "document" : new_doc
        }
    # if tool message is missing then fallback to previous state
    return state


@tool
def save_doc(filename: str, doc_state: Annotated[str, InjectedState("document")]) -> str:
    """
    Save the current document to a text file and complete the drafting process.
    
    This tool writes the current document content to a specified text file.
    It automatically appends the .txt extension if not provided and handles
    file encoding properly. The function returns a JSON status message
    indicating success or failure.
    
    Args:
        filename (str): Name of the text file to save the document to.
                       The .txt extension will be added automatically if missing.
        doc_state (str): The current document content (automatically injected
                        from the agent state).
                        
    Returns:
        str: A JSON string containing:
            - status: "success" or "failed" indicating the operation result
            - message: Detailed message about the save operation or error details
    """
    try:
        if not filename.endswith(".txt"):
            filename = f"{filename}.txt"
            
        with open(filename, "w", encoding ='utf-8') as file:   
            file.write(doc_state)
            
        return json.dumps({
            "status": "success",
            "message": f'Document saved successfully o {filename}'
        })
    
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "message": f'Error saving document: {str(e)}'
        })
        
tools = [update_doc, save_doc]
model  = ChatOpenAI(model="gpt-4.1-nano").bind_tools(tools=tools)
        
def agent(state: AgentState) -> AgentState:
    """ """
    system_prompt = SystemMessage(content=f"""
                                  You are Cuddles, a helpful writing assistant. You are going to help the user update and modify documents.
                                  - If the user wants to update or modify content, use the "update_doc' tool with the complete updated document.
                                  - If the user wants to save and finish, you need to use the 'save_doc' tool.
                                  - Make sure to always show the current document state after modifications like this:
                                  
                                  The current document content is: <document_content here>
                                  """)
    
    if not state["messages"]:
        # First interaction
        intro_message = AIMessage(content="I'm ready to help you update a document. What would you like to create?")
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content=user_input)
        
        # Send to model with system prompt
        all_messages = [system_prompt, intro_message, user_message]
        
        response = model.invoke(all_messages)
        
        __print_resp_and_tc(response)
        
        return {
            **state,
            "messages": [intro_message, user_message, response]
        }
    else:
        # Subsequent interactions
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content=user_input)
        
        # Send all existing messages plus new one
        all_messages = [system_prompt] + list(state["messages"]) + [user_message]
        
        response = model.invoke(all_messages)
        
        __print_resp_and_tc(response)
        
        return {
            **state,
            "messages": [user_message, response]
        }
        
def __print_resp_and_tc(response: BaseMessage) -> None:
    print(f'\nAI: {response.content}')
    
    # print tool calls by the llm
    if isinstance(response, AIMessage) and hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc["name"] for tc in response.tool_calls]}")



