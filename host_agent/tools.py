"""Host Agent tools for A2A communication with Notion and ElevenLabs agents.

Follows a generic delegation pattern. The host agent has one tool to delegate
any task to a named child agent.
"""

import asyncio
import os
import sys
from typing import Dict

from config import ELEVENLABS_AGENT_A2A_URL, NOTION_AGENT_A2A_URL
from host_agent.remote_connections import RemoteConnections

# Add project root to the Python path to resolve module import errors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mapping of agent names to their URLs. In a more dynamic system,
# this could come from a service discovery mechanism.
AGENT_URL_MAP: Dict[str, str] = {
    "notion_agent": NOTION_AGENT_A2A_URL,
    "elevenlabs_agent": ELEVENLABS_AGENT_A2A_URL,
}


async def delegate_task(agent_name: str, task_description: str) -> str:
    """Delegates a task to a specified child agent via A2A protocol.

    Args:
        agent_name: The logical name of the target agent (e.g., 'notion_agent').
        task_description: A detailed description of the task for the child agent to perform.

    Returns:
        The result from the child agent, or an error message.
    """
    if agent_name not in AGENT_URL_MAP:
        return f"Error: Agent '{agent_name}' is not a known agent. Available agents are: {list(AGENT_URL_MAP.keys())}"

    agent_url = AGENT_URL_MAP[agent_name]
    remote_connections = await RemoteConnections.create(
        timeout=60.0
    )  # Increased timeout for complex tasks

    try:
        # The task_description is passed directly as the query to the child agent.
        # This allows the host agent to give rich, detailed instructions.
        result = await remote_connections.invoke_agent(agent_url, task_description)

        if isinstance(result, dict):
            if result.get("error"):
                return f"Error from {agent_name}: {result['error']}"
            elif result.get("result"):
                return result["result"]
            else:
                return f"Error: Unexpected response format from {agent_name}"
        else:
            return "Error: Invalid response format from RemoteConnections"

    except Exception as e:
        return f"Error delegating task to {agent_name}: {str(e)}"
    finally:
        await remote_connections.close()


def delegate_task_sync(agent_name: str, task_description: str) -> str:
    """
    Synchronous wrapper for delegate_task to be used as an ADK tool.

    This function handles running the async 'delegate_task' function from
    a synchronous context, which is required for ADK tools. It intelligently
    handles cases where an asyncio event loop is already running.

    Args:
        agent_name: The logical name of the target agent.
        task_description: A detailed description of the task.

    Returns:
        The result from the child agent, or an error message.
    """
    try:
        # This pattern handles both scenarios: running within an existing
        # event loop (like in some web frameworks or notebooks) or running
        # in a standard synchronous environment.
        try:
            asyncio.get_running_loop()
            # If inside an event loop, run the async code in a separate thread
            # to avoid "RuntimeError: cannot be called from a running event loop".
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, delegate_task(agent_name, task_description)
                )
                return future.result(
                    timeout=90
                )  # Generous timeout for orchestrated tasks
        except RuntimeError:
            # No running event loop, so we can safely use asyncio.run()
            return asyncio.run(delegate_task(agent_name, task_description))
    except Exception as e:
        return f"Error in sync delegation wrapper: {str(e)}"
