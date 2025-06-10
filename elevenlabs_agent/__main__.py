import logging
import os

import click
import uvicorn
from dotenv import load_dotenv

# ADK imports
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# A2A server imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

# Local agent imports
# This import assumes elevenlabs_agent/agent_executor.py exists and defines ElevenLabsADKAgentExecutor
from elevenlabs_agent.agent_executor import ElevenLabsADKAgentExecutor
from elevenlabs_agent.agent import create_elevenlabs_agent # To create the actual ADK agent

# Load environment variables from .env file
load_dotenv()

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    "host",
    default=os.getenv("A2A_ELEVENLABS_HOST", "localhost"),
    show_default=True,
    help="Host for the ElevenLabs agent server."
)
@click.option(
    "--port",
    "port",
    default=int(os.getenv("A2A_ELEVENLABS_PORT", 8003)), # Adjusted default port
    show_default=True,
    type=int,
    help="Port for the ElevenLabs agent server."
)
def main(host: str, port: int) -> None:
    """Runs the ElevenLabs ADK Agent as an A2A service."""

    # The ELEVENLABS_API_KEY is used by the MCP server started by the agent.
    # We should ensure it's available in the environment where the agent runs.
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.warning(
            "ELEVENLABS_API_KEY environment variable not set. "
            "The ElevenLabs MCP server might fail to authenticate."
        )
    # ADK_MODEL and other ADK-related GOOGLE_API_KEY checks are implicitly handled
    # by the ADK agent itself if it uses Google LLMs. For ElevenLabs, the primary
    # key is ELEVENLABS_API_KEY for the MCP tool.

    # 1. Define AgentCard for ElevenLabs
    eleven_skill = AgentSkill(
        id="text_to_speech",
        name="Convert text to speech",
        description="Takes input text and returns an audio file of the spoken text using ElevenLabs.",
        tags=["tts", "audio", "speech", "elevenlabs"],
        examples=["Say 'Hello, world!'", "Convert the following to speech: Today is a wonderful day."],
    )

    agent_card = AgentCard(
        name="ElevenLabs TTS Agent",
        description="Provides text-to-speech services using ElevenLabs.",
        url=f"http://{host}:{port}/", # URL is dynamically set here
        version="1.0.0",
        defaultInputModes=["text"], # Agent primarily takes text
        defaultOutputModes=["text","audio"], # Agent primarily outputs audio (e.g., audio/mpeg)
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False), # TTS is typically not streaming response
        skills=[eleven_skill],
    )

    try:
        # 2. Create the actual ADK LlmAgent
        adk_llm_agent = create_elevenlabs_agent()

        # 3. Initialize the ADK Runner
        runner = Runner(
            app_name=agent_card.name, # Use name from AgentCard for Runner
            agent=adk_llm_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        # 4. Instantiate the AgentExecutor, passing the runner and card
        agent_executor = ElevenLabsADKAgentExecutor(agent=adk_llm_agent, agent_card=agent_card)

    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs Agent components: {e}", exc_info=True)
        return

    # 5. Set up the A2A request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    # 6. Create the A2A Starlette application
    # The agent_card here is the one defined locally, now fully configured.
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info(f"Starting ElevenLabs Agent server on http://{host}:{port}")
    logger.info(f"Agent Name: {agent_card.name}, Version: {agent_card.version}")
    if agent_card.skills:
        for skill in agent_card.skills:
            logger.info(f"  Skill: {skill.name} (ID: {skill.id}, Tags: {skill.tags})")

    # Run the Uvicorn server
    uvicorn.run(a2a_app.build(), host=host, port=port)


if __name__ == "__main__":
    main() 