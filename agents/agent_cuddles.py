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
    Save the current document to a markdown file and complete the drafting process.
    
    This tool writes the current document content to a specified md file.
    It automatically appends the .md extension if not provided and handles
    file encoding properly. The function returns a JSON status message
    indicating success or failure.
    
    Args:
        filename (str): Name of the markdown file to save the document to.
                       The .md extension will be added automatically if missing.
        doc_state (str): The current document content (automatically injected
                        from the agent state).
                        
    Returns:
        str: A JSON string containing:
            - status: "success" or "failed" indicating the operation result
            - message: Detailed message about the save operation or error details
    """
    try:
        if not filename.endswith(".md"):
            filename = f"{filename}.md"
            
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

@tool
def show_current_doc(cur_doc: Annotated[str, InjectedState("document")]) -> str:
    """Returns the current document content as a string. No arguments required."""
    return cur_doc

tools = [update_doc, save_doc, show_current_doc]
model  = ChatOpenAI(model="gpt-4.1-nano").bind_tools(tools=tools)


def agent(state: AgentState) -> AgentState:
    """
    Main agent node that handles user interactions and AI responses.
    
    This function has two main roles:
    1.  When following a tool execution, it calls the AI model to generate a response 
        to the tool's result without asking for user input.
    2.  Otherwise, it prompts the user for their next command and then calls the AI.
        
    This dual role allows the AI to both react to its own actions (like showing an
    updated document) and respond to new user instructions in a continuous loop.
    """
    system_prompt = SystemMessage(content=f"""
You are Cuddles 🧸, a professional document planning assistant with a friendly personality.

CORE MISSION:
Help users create, edit, and organize documents for projects, workflows, lists, notes, and planning tasks.

IMPORTANT RULES:
- When a user asks to CREATE, WRITE, MAKE, or BUILD any document/plan/content, you MUST immediately use the "update_doc" tool with the complete content.
- You must write to the document in a professional tone that guides the user to achieve their goal. Do not add filler conclusions or ask the user questions at the end of the document.
- When a user asks to SAVE, FINISH, or EXPORT the document, use the "save_doc" tool to complete the session.
- Do NOT just talk about creating content - actually create it using the tools!
- Be encouraging and professional while maintaining a helpful, friendly tone.

- ALWAYS SHOW DOCUMENT STATE AFTER UPDATING THE DOC, EVEN AFTER CONSECUTIVE DOC UPDATE TOOL CALLS.
- After every successful update_doc tool call, you MUST display the updated document to the user in this exact format:

📄 **Updated Document:**
```
[complete document content here]
```

This helps users see exactly what was created or modified in a clean, readable format.

Remember: Users appreciate seeing their documents clearly formatted and updated in real-time. Always show the complete content after making changes.
""")
    
    # Check if the last message is a tool result.
    # If so, the model should respond to the tool execution without asking for new user input.
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        # AI's turn to respond to the tool result
        all_messages = [system_prompt] + list(state["messages"])
        response = model.invoke(all_messages)
        __print_resp_and_tc(response)
        # The new message will be appended to the state by `add_messages`
        return {**state, "messages": [response]}

    # This is the first turn, we need to greet and get user input
    if not state["messages"]:
        intro_message = AIMessage(content="Hi! I'm Cuddles 🧸, your document planning assistant. I'm here to help you create, organize, and manage your projects, plans, and documents. What would you like to work on today?")
        user_input = input("\n💭 What would you like to create or work on? (projects, lists, plans, notes, etc.)\n→ ")
        user_message = HumanMessage(content=user_input)
        
        all_messages = [system_prompt, intro_message, user_message]
        response = model.invoke(all_messages)
        __print_resp_and_tc(response)
        return {**state, "messages": [intro_message, user_message, response]}

    # This is a subsequent turn where we need user input
    else:
        user_input = input("\n✏️  What would you like to do next? (edit, view doc, add content, save, etc.)\n→ ")
        user_message = HumanMessage(content=user_input)
        
        all_messages = [system_prompt] + list(state["messages"]) + [user_message]
        response = model.invoke(all_messages)
        __print_resp_and_tc(response)
        return {**state,"messages": [user_message, response]}
        
def __print_resp_and_tc(response: BaseMessage) -> None:
    print(f'\nAI: {response.content}')
    
    # print tool calls by the llm
    if isinstance(response, AIMessage) and hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc["name"] for tc in response.tool_calls]}")

def should_continue_after_tools(state: AgentState) -> str:
    """ Route after tool execution based on which tool was called."""
    messages = state["messages"]
    
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage):
        tool_name = last_message.name
        if tool_name == "update_doc":
            return "process_update"
        elif tool_name == "save_doc":
            return "end"
        elif tool_name == "show_current_doc":
            return 'back_to_agent'
        
    return "back_to_agent"


def should_continue_after_agent(state: AgentState) -> str:
    """ Determine where to route from the agent node."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If AI wants to use tools, go to tools. Otherwise keep conversation going.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue_to_tools"
    else:
        return "continue_conversation" 

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("update", process_doc_update)


graph.add_conditional_edges("agent", should_continue_after_agent, {
    "continue_to_tools": "tools",
    "continue_conversation": "agent"
})
graph.add_edge("update", "agent")

graph.add_conditional_edges("tools", should_continue_after_tools, {
    "end": END,
    "process_update" : "update",
    "back_to_agent": "agent"
})

graph.set_entry_point("agent")

app = graph.compile()


def print_current_state(step_data: dict) -> None:
    """Prints the current state of the document and messages in a readable format."""
    if "messages" in step_data and step_data["messages"]:
        last_msg = step_data["messages"][-1]
        
        if isinstance(last_msg, HumanMessage):
            print(f"👤 User: {last_msg.content}")
        elif isinstance(last_msg, AIMessage):
            print(f"🤖 AI: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                tools_used = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
                print(f"   🔧 Tools: {', '.join(tools_used)}")
        elif isinstance(last_msg, ToolMessage):
            print(f"⚙️  Tool: {last_msg.name} completed")


def run_cuddles_agent() -> None:
    """Run the Cuddles document assistant with improved UI formatting."""
    
    # Welcome header
    print("\n" + "="*60)
    print("🧸 CUDDLES - Document Planning Assistant 🧸".center(60))
    print("="*60)
    print("✨ Welcome! I'm here to help you create and edit documents.")
    print("💡 I can help with project plans, lists, notes, and more!")
    print("💾 Say 'save' when you're ready to save your document.")
    print("-"*60)
    
    # Initialize state
    state: AgentState = {
        "messages": [],
        "document": "Document is empty"
    }
    
    try:
        # Main interaction loop
        for step in app.stream(state):
            
            if "messages" in step:
                print_current_state(step)  # type: ignore
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Session interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
    finally:
        # Closing message
        print("\n" + "="*60)
        print("🏁 SESSION COMPLETED".center(60))
        print("="*60)
        print("✅ Thank you for using Cuddles!")
        print("👋 Have a great day!")
        print("="*60)


def print_simple_step(step_data: dict) -> None:
    """Print a simplified view of each step for debugging."""
    if "messages" in step_data and step_data["messages"]:
        last_msg = step_data["messages"][-1]
        
        if isinstance(last_msg, HumanMessage):
            print(f"👤 User: {last_msg.content[:80]}...")
        elif isinstance(last_msg, AIMessage):
            print(f"🤖 AI: {last_msg.content[:80]}...")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                tools_used = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
                print(f"   🔧 Tools: {', '.join(tools_used)}")
        elif isinstance(last_msg, ToolMessage):
            print(f"⚙️  Tool: {last_msg.name} completed")


if __name__ == "__main__":
    run_cuddles_agent()




