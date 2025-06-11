"""Host Agent implementation using ADK with a generic A2A delegation tool."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from config import GOOGLE_API_KEY
from host_agent.prompt import HOST_PROMPT
from host_agent.tools import delegate_task_sync


def create_host_agent() -> Agent:
    """Creates an ADK agent that orchestrates child agents via a generic delegation tool.

    The Host Agent uses a single, powerful tool to delegate tasks to the
    ElevenLabs and Notion agents over HTTP. It can coordinate complex workflows
    by sending detailed instructions to the appropriate child agent.

    Returns:
        Agent: An ADK Agent configured with the generic delegation tool and an orchestration prompt.
    """
    return Agent(
        name="host_agent_orchestrator",
        model=LiteLlm(model="gemini/gemini-2.0-flash", api_key=GOOGLE_API_KEY),
        description="A master orchestrator that delegates tasks to specialized child agents (Notion, ElevenLabs) using a generic A2A communication tool.",
        instruction=HOST_PROMPT,
        tools=[
            delegate_task_sync,
        ],
    )


# Required for ADK discovery
root_agent = create_host_agent()
