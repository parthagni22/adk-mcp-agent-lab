"""
Streamlit UI for the Host Agent using embedded ADK Runner pattern.

This UI directly integrates with the Host Agent's ADK Runner in the same process,
following the embedded runner pattern from @adk-ui.mdc.
"""

import os
import sys

# Add project root to the Python path to resolve module import errors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import asyncio
from typing import Any, Dict, List
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from google.genai import types
import traceback
import uuid

# Import the host agent creation logic
from host_agent.agent import create_host_agent

# App configuration
APP_NAME = "host_agent_ui"
USER_ID = "streamlit_user"

# --- Service and Runner Initialization ---

@st.cache_resource
def get_adk_runner() -> Runner:
    """
    Initializes and caches the ADK Runner using Streamlit's resource caching.
    
    This function is decorated with @st.cache_resource, which ensures that:
    - A single, persistent Runner instance is created per user session
    - The Runner maintains the agent's context and memory across Streamlit reruns
    - The function is only executed once per session, not on every interaction
    
    Returns:
        Runner: The cached ADK Runner instance with persistent session state
    """
    print("🔧 Creating new ADK Runner instance (this should only appear once per session)")
    
    session_service = InMemorySessionService()
    host_agent = create_host_agent()
    return Runner(
        agent=host_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

@st.cache_resource
def initialize_adk_session():
    """
    Creates the ADK session once per user session using resource caching.

    This function is decorated with @st.cache_resource, which guarantees
    that the code inside runs only once. It retrieves the cached ADK runner
    and uses it to create a single, persistent ADK session. This prevents
    the session context from being reset on every user interaction.

    Returns:
        bool: True when the session is created successfully.
    """
    print(f"--- This function is a placeholder and its logic has been moved into run_agent_logic ---")
    return True

# --- Agent Logic ---

async def run_agent_logic(prompt: str, session_id: str) -> Dict[str, Any]:
    """
    Instantiates and runs the Host Agent, returning structured results.
    
    Args:
        prompt: User's input message
        
    Returns:
        Dictionary containing final response, tool calls, and any artifacts (like audio URLs)
    """
    try:
        # Retrieve the persistent runner instance from the cache.
        runner = get_adk_runner()
        
        # Ensure ADK session exists ONCE per streamlit session
        if 'adk_session_initialized' not in st.session_state:
            try:
                await runner.session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id
                )
                print(f"✅ ADK session created: {session_id}")
                st.session_state.adk_session_initialized = True
            except Exception:
                # Session might already exist in a rare case, but we can proceed.
                st.session_state.adk_session_initialized = True
        
        # Track results
        tool_calls = []
        tool_responses = []
        final_response = ""
        audio_url = None
        
        # Execute the runner and process events (using the dynamic session_id)
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,  # Use dynamic session ID for user isolation
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Handle function calls (tool calls)
                    if part.function_call:
                        tool_calls.append({
                            'name': part.function_call.name,
                            'args': part.function_call.args
                        })
                    
                    # Handle function responses (tool responses)
                    elif part.function_response:
                        response_data = part.function_response.response
                        tool_responses.append({
                            'name': part.function_response.name,
                            'response': response_data
                        })
                        
                        # Check for audio URL in ElevenLabs responses
                        if (isinstance(response_data, dict) and 
                            'response' in response_data and 
                            isinstance(response_data['response'], dict)):
                            inner_response = response_data['response']
                            if 'audio_url' in inner_response:
                                audio_url = inner_response['audio_url']
                        elif isinstance(response_data, dict) and 'audio_url' in response_data:
                            audio_url = response_data['audio_url']
            
            # Handle final response
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = "".join([p.text for p in event.content.parts if p.text])
                elif event.actions and event.actions.escalate:
                    final_response = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        return {
            'final_response': final_response,
            'tool_calls': tool_calls,
            'tool_responses': tool_responses,
            'audio_url': audio_url,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Error running agent: {str(e)}")
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
        st.session_state.session_id = f"session-{uuid.uuid4()}"
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []


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
        page_title="Host Agent Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Get the runner instance (this will be cached after the first run)
    get_adk_runner()
    
    # Initialize session state for UI elements
    initialize_session_state()
    
    # Main UI
    st.title("🤖 Host Agent Assistant")
    st.markdown("Chat with the Host Agent that can search Notion and create audio using ElevenLabs.")
    
    # Sidebar with session info
    with st.sidebar:
        st.header("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:13]}...")
        
        # Show that the runner is cached and persistent
        st.info("🧠 **Agent Memory**: Context is preserved for this session.")
        
        if st.button("🔄 New Session"):
            # Clear all session state which triggers re-initialization
            st.session_state.clear()
            st.rerun()
        
        # Display audio files
        if st.session_state.audio_files:
            st.header("🎵 Generated Audio")
            for i, audio_url in enumerate(st.session_state.audio_files):
                st.audio(audio_url, format="audio/mp3")
    
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
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 Agent is thinking and coordinating with child agents..."):
                # Use asyncio.run to bridge sync UI to async ADK logic
                result = asyncio.run(run_agent_logic(prompt, st.session_state.session_id))
            
            # Display final response
            if result['final_response']:
                st.write(result['final_response'])
            
            # Show tool interactions
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