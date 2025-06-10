"""
Streamlit UI for the Host Agent using A2A polling architecture.

This UI communicates with the Host Agent as a decoupled service via the A2A SDK,
following the polling pattern from host_agent/test_client.py.
This is in contrast to ui/app.py which uses the embedded ADK Runner pattern.
"""

import streamlit as st
import asyncio
import httpx
import os
import traceback
import uuid
from uuid import uuid4
from typing import Any, Dict, List
from dotenv import load_dotenv

from a2a.client import A2AClient
from a2a.types import (
    GetTaskRequest,
    GetTaskResponse,
    GetTaskSuccessResponse,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    TaskQueryParams,
)

# Load environment variables
load_dotenv()

# Configuration
AGENT_URL = os.getenv("HOST_AGENT_A2A_URL", "http://localhost:8001")
MAX_RETRIES = 15  # Longer timeout for orchestration workflows
RETRY_DELAY = 3   # Seconds between polling attempts
TIMEOUT = 90.0    # HTTP timeout for A2A operations

# --- A2A Client Functions ---

def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> Dict[str, Any]:
    """Create A2A send message payload with proper format.
    
    Args:
        text: The message text to send
        task_id: Optional task ID to associate with the message
        context_id: Optional context ID for conversation context
        
    Returns:
        Dict: Properly formatted A2A message payload
    """
    # Use stored context ID for conversation continuity if available
    if context_id is None and 'current_context_id' in st.session_state:
        context_id = st.session_state.current_context_id
        print(f"🔍 DEBUG: Using stored contextId for continuity: {context_id}")
    
    payload: Dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'text': text}], 
            'messageId': uuid4().hex,
        },
    }
    if task_id:
        payload['message']['taskId'] = task_id
    if context_id:
        payload['message']['contextId'] = context_id
    return payload


async def send_message_to_agent(client: A2AClient, text: str, context_id: str | None = None) -> str | None:
    """Send a message to the agent and return the task ID.
    
    Args:
        client: The A2A client to use
        text: The message text to send
        
    Returns:
        str | None: The task ID if successful, None otherwise
    """
    try:
        send_payload = create_send_message_payload(text=text)
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_payload))
        
        print("🔍 DEBUG: Sending request to agent...")
        send_response: SendMessageResponse = await client.send_message(request)
        
        # Debug logging to console
        print(f"🔍 DEBUG: Response type: {type(send_response)}")
        if hasattr(send_response, 'model_dump_json'):
            print(f"🔍 DEBUG: Response JSON: {send_response.model_dump_json(exclude_none=True)}")
        
        # Handle union type wrapper - access the actual response
        if hasattr(send_response, 'root'):
            actual_response = send_response.root
            print(f"🔍 DEBUG: Found root attribute, type: {type(actual_response)}")
        else:
            actual_response = send_response
            print("🔍 DEBUG: No root attribute, using response directly")
        
        # Check if we have result attribute on the actual response
        if not hasattr(actual_response, 'result'):
            st.error('Response does not have result attribute')
            return None

        # Extract task ID and immediate response from agent's reply
        agent_reply_data = actual_response.result  # type: ignore
        print(f"🔍 DEBUG: agent_reply_data type: {type(agent_reply_data)}")
        
        # Check for immediate text response in parts (handle safely)
        immediate_text = None
        if hasattr(agent_reply_data, 'parts'):
            parts = getattr(agent_reply_data, 'parts', [])
            if parts:
                for part in parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        immediate_text = getattr(part.root, 'text', None)
                        if immediate_text:
                            print(f"🔍 DEBUG: Found immediate text response: {immediate_text[:100]}...")
                            break

        # Extract task ID from the message (matching test_client.py pattern)
        extracted_task_id: str | None = None

        # Handle both Pydantic models and dict responses
        print(f"🔍 DEBUG: Checking for taskId attribute: {hasattr(agent_reply_data, 'taskId')}")
        if hasattr(agent_reply_data, 'taskId'):
            task_id_value = getattr(agent_reply_data, 'taskId', None)
            print(f"🔍 DEBUG: taskId value from attribute: {task_id_value}")
            if isinstance(task_id_value, str):
                extracted_task_id = task_id_value
        
        if not extracted_task_id and isinstance(agent_reply_data, dict):
            task_id_value = agent_reply_data.get('taskId')
            print(f"🔍 DEBUG: taskId value from dict: {task_id_value}")
            if isinstance(task_id_value, str):
                extracted_task_id = task_id_value

        print(f"🔍 DEBUG: Final extracted_task_id: {extracted_task_id}")

        if not extracted_task_id:
            st.error("Could not extract taskId from the agent's reply")
            return None

        # Store immediate response if we found one
        if immediate_text:
            if 'immediate_responses' not in st.session_state:
                st.session_state.immediate_responses = {}
            st.session_state.immediate_responses[extracted_task_id] = immediate_text
            print(f"🔍 DEBUG: Stored immediate response for task {extracted_task_id}")

        # Store contextId for conversation continuity
        if hasattr(agent_reply_data, 'contextId'):
            context_id = getattr(agent_reply_data, 'contextId', None)
            if context_id:
                st.session_state.current_context_id = context_id
                print(f"🔍 DEBUG: Stored contextId for continuity: {context_id}")

        return extracted_task_id
        
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


async def poll_for_task_completion(client: A2AClient, task_id: str) -> Dict[str, Any]:
    """Poll for task completion and return structured results.
    
    Args:
        client: The A2A client to use
        task_id: The task ID to poll for
        
    Returns:
        Dict: Structured results containing final response, tool calls, etc.
    """
    results = {
        'final_response': '',
        'tool_calls': [],
        'tool_responses': [],
        'audio_url': None,
        'success': False
    }
    

    # Check for immediate response first
    if ('immediate_responses' in st.session_state and 
        task_id in st.session_state.immediate_responses):
        immediate_text = st.session_state.immediate_responses[task_id]
        print(f"🔍 DEBUG: Using immediate response for task {task_id}")
        results['final_response'] = immediate_text
        results['success'] = True
        # Clean up the immediate response
        del st.session_state.immediate_responses[task_id]
        return results
    
    try:
        task_status = "unknown"
        print(f"🔍 DEBUG: No immediate response found, starting polling for task {task_id}")
        
        for attempt in range(MAX_RETRIES):
            get_request = GetTaskRequest(id=str(uuid4()), params=TaskQueryParams(id=task_id))
            get_response: GetTaskResponse = await client.get_task(get_request)

            # Handle union type wrapper for get_response
            if hasattr(get_response, 'root'):
                actual_get_response = get_response.root
            else:
                actual_get_response = get_response
                
            if hasattr(actual_get_response, 'result'):
                actual_task_result = getattr(actual_get_response, 'result', None)
                if not actual_task_result:
                    continue
            else:
                continue
                
            if actual_task_result and hasattr(actual_task_result, 'status'):
                task_status = actual_task_result.status.state
                
                if task_status in ["completed", "failed"]:
                    if task_status == "completed" and hasattr(actual_task_result, 'artifacts') and actual_task_result.artifacts:
                        # Process artifacts to extract results
                        final_text_parts = []
                        
                        for artifact_item in actual_task_result.artifacts:
                            if isinstance(artifact_item, dict):
                                parts_list = artifact_item.get('parts')
                                if isinstance(parts_list, list):
                                    for part_data in parts_list:
                                        if isinstance(part_data, dict):
                                            text = part_data.get('text')
                                            audio_url = part_data.get('audio_url')
                                            
                                            if text:
                                                final_text_parts.append(text)
                                            if audio_url and not results['audio_url']:
                                                results['audio_url'] = audio_url
                        
                        results['final_response'] = '\n'.join(final_text_parts) if final_text_parts else "Task completed successfully."
                        results['success'] = True
                        
                    elif task_status == "failed":
                        error_msg = actual_task_result.status.message if hasattr(actual_task_result.status, 'message') and actual_task_result.status.message else "Task failed"
                        results['final_response'] = f"❌ Task Failed: {error_msg}"
                        results['success'] = False
                    
                    return results
            
            # If not completed, wait and retry
            if attempt < MAX_RETRIES - 1 and task_status not in ["completed", "failed"]:
                await asyncio.sleep(RETRY_DELAY)
            elif task_status in ["completed", "failed"]:
                break
        
        # If we get here, max retries reached
        results['final_response'] = f"⏰ Timeout: Task did not complete after {MAX_RETRIES} attempts"
        results['success'] = False
        return results
        
    except Exception as e:
        results['final_response'] = f"Error polling for task completion: {str(e)}"
        results['success'] = False
        return results


@st.cache_resource
def get_a2a_client() -> None:
    """Initialize A2A client connection placeholder.
    
    Note: Due to async requirements, actual client creation happens in run_agent_logic_a2a.
    This function serves as a placeholder for Streamlit's caching mechanism.
    """
    print("🔧 A2A Client placeholder initialized (actual client created per request)")
    return None


async def create_a2a_client() -> A2AClient | None:
    """Create A2A client connection asynchronously.
    
    Returns:
        A2AClient | None: The A2A client or None if connection failed
    """
    try:
        print(f"🔍 DEBUG: Connecting to agent at {AGENT_URL}")
        httpx_client = httpx.AsyncClient(timeout=TIMEOUT)
        client = await A2AClient.get_client_from_agent_card_url(httpx_client, AGENT_URL)
        print("🔍 DEBUG: A2A client created successfully")
        return client
    except httpx.ConnectError as e:
        st.error(f"❌ Connection error: Could not connect to agent at {AGENT_URL}. Ensure the server is running.")
        st.info("To start the host agent server: `python -m host_agent --port 8001`")
        st.code(str(e), language="python")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


# --- Agent Logic ---

async def run_agent_logic_a2a(prompt: str) -> Dict[str, Any]:
    """
    Send a message to the Host Agent via A2A and poll for results.
    
    Args:
        prompt: User's input message
        
    Returns:
        Dictionary containing final response, tool calls, and any artifacts (like audio URLs)
    """
    try:
        # Create A2A client
        client = await create_a2a_client()
        if not client:
            return {
                'final_response': "❌ Failed to connect to Host Agent service",
                'tool_calls': [],
                'tool_responses': [],
                'audio_url': None,
                'success': False
            }
        
        # Step 1: Send message and get task ID
        print("🔍 DEBUG: Step 1 - Sending message to agent")
        task_id = await send_message_to_agent(client, prompt)
        if not task_id:
            print("🔍 DEBUG: Failed to get task ID from agent")
            return {
                'final_response': "❌ Failed to send message to agent",
                'tool_calls': [],
                'tool_responses': [],
                'audio_url': None,
                'success': False
            }
        
        st.info(f"📤 Task submitted with ID: {task_id}")
        print(f"🔍 DEBUG: Got task ID: {task_id}")
        
        # Step 2: Poll for completion
        print("🔍 DEBUG: Step 2 - Starting polling")
        with st.spinner("🔄 Polling for task completion..."):
            results = await poll_for_task_completion(client, task_id)
        
        return results
        
    except Exception as e:
        st.error(f"Error in A2A communication: {str(e)}")
        traceback.print_exc()
        return {
            'final_response': f"An error occurred: {str(e)}",
            'tool_calls': [],
            'tool_responses': [],
            'audio_url': None,
            'success': False
        }


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"a2a-session-{uuid.uuid4()}"
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []
    if 'current_context_id' not in st.session_state:
        st.session_state.current_context_id = None


def display_tool_calls(tool_calls: List[Dict[str, Any]]):
    """Display tool calls in an expandable section."""
    if tool_calls:
        with st.expander(f"🛠️ Tool Calls ({len(tool_calls)})", expanded=False):
            for i, call in enumerate(tool_calls):
                st.code(f"Tool: {call['name']}\nArguments: {call['args']}", language="python")


def display_tool_responses(tool_responses: List[Dict[str, Any]]):
    """Display tool responses in an expandable section."""
    if tool_responses:
        with st.expander(f"⚡ Tool Responses ({len(tool_responses)})", expanded=False):
            for i, response in enumerate(tool_responses):
                st.write(f"**{response['name']}:**")
                if isinstance(response['response'], dict):
                    st.json(response['response'])
                else:
                    st.text(str(response['response']))


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Host Agent Assistant (A2A)",
        page_icon="🌐",
        layout="wide"
    )
    
    # Initialize the cached A2A client
    get_a2a_client()
    
    # Initialize session state for UI elements
    initialize_session_state()
    
    # Main UI
    st.title("🌐 Host Agent Assistant (A2A Architecture)")
    st.markdown("Chat with the Host Agent via A2A protocol that can search Notion and create audio using ElevenLabs.")
    
    # Sidebar with session info and architecture notes
    with st.sidebar:
        st.header("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:13]}...")
        
        # Architecture information
        st.info("🌐 **A2A Architecture**: This UI communicates with the Host Agent as a decoupled service via HTTP/A2A protocol.")
        st.info(f"🔗 **Agent URL**: {AGENT_URL}")
        
        if st.button("🔄 New Session"):
            # Clear all session state which triggers re-initialization
            st.session_state.clear()
            st.rerun()
        
        # Display audio files
        if st.session_state.audio_files:
            st.header("🎵 Generated Audio")
            for i, audio_url in enumerate(st.session_state.audio_files):
                st.audio(audio_url, format="audio/mp3")
        
        # Comparison with embedded UI
        st.header("🔀 Architecture Comparison")
        st.write("**Embedded UI** (`ui/app.py`):")
        st.write("- Direct ADK Runner integration")
        st.write("- Same process, shared memory")
        st.write("- Immediate event streaming")
        
        st.write("**A2A UI** (this app):")
        st.write("- HTTP-based communication")
        st.write("- Decoupled services")
        st.write("- Polling-based task status")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show tool interactions if present
            if "tool_calls" in message:
                display_tool_calls(message["tool_calls"])
            if "tool_responses" in message:
                display_tool_responses(message["tool_responses"])
    
    # Chat input
    if prompt := st.chat_input("Ask me to search Notion or create audio..."):
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with agent via A2A
        with st.chat_message("assistant"):
            with st.spinner("🤔 Agent is processing your request via A2A..."):
                # Use asyncio.run to bridge sync UI to async A2A logic
                result = asyncio.run(run_agent_logic_a2a(prompt))
            
            # Display final response
            if result['final_response']:
                st.write(result['final_response'])
            
            # Show tool interactions (if available in A2A response)
            display_tool_calls(result['tool_calls'])
            display_tool_responses(result['tool_responses'])
            
            # Handle audio if present
            if result['audio_url']:
                st.audio(result['audio_url'], format="audio/mp3")
                # Add to session audio files
                if result['audio_url'] not in st.session_state.audio_files:
                    st.session_state.audio_files.append(result['audio_url'])
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": result['final_response'],
                "tool_calls": result['tool_calls'],
                "tool_responses": result['tool_responses']
            }
            st.session_state.conversation_history.append(assistant_message)


if __name__ == "__main__":
    main() 