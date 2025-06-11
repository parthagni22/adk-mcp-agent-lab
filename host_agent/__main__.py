"""Host Agent A2A Service Entry Point."""

import logging
import os

import click
import uvicorn

# A2A server imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

# ADK imports
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from host_agent.agent import create_host_agent

# Local agent imports
from host_agent.agent_executor import HostADKAgentExecutor

# Load environment variables from .env file
load_dotenv()

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    "host",
    default=os.getenv("A2A_HOST_HOST", "0.0.0.0"),
    show_default=True,
    help="Host for the Host agent server.",
)
@click.option(
    "--port",
    "port",
    default=int(os.getenv("A2A_HOST_PORT", 8001)),
    show_default=True,
    type=int,
    help="Port for the Host agent server.",
)
def main(host: str, port: int) -> None:
    """Runs the Host ADK Agent as an A2A service."""

    # Check for required API keys
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning(
            "GOOGLE_API_KEY environment variable not set. "
            "The Host agent might fail to initialize."
        )

    # Define AgentCard for Host Agent
    orchestration_skill = AgentSkill(
        id="orchestrate_workflows",
        name="Orchestrate Multi-Agent Workflows",
        description="Coordinates Notion information retrieval and ElevenLabs text-to-speech generation.",
        tags=["orchestration", "workflow", "coordination", "multi-agent"],
        examples=[
            "Search Notion for project updates and convert to speech",
            "Find meeting notes and generate audio summary",
            "Retrieve documentation and create audio version",
        ],
    )

    agent_card = AgentCard(
        name="Host Agent Orchestrator",
        description="Orchestrates Notion and ElevenLabs agents via A2A protocol for information retrieval and text-to-speech workflows.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[orchestration_skill],
    )

    try:
        # Create the actual ADK Agent
        adk_agent = create_host_agent()

        # Initialize the ADK Runner (following official ADK pattern)
        runner = Runner(
            app_name=agent_card.name,
            agent=adk_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        # Instantiate the AgentExecutor with the runner
        agent_executor = HostADKAgentExecutor(
            agent=adk_agent, agent_card=agent_card, runner=runner
        )

    except Exception as e:
        logger.error(f"Failed to initialize Host Agent components: {e}", exc_info=True)
        return

    # Set up the A2A request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    # Create the A2A Starlette application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info(f"🌟 Starting Host Agent A2A Server on port {port}")
    logger.info(f"Agent Name: {agent_card.name}, Version: {agent_card.version}")
    if agent_card.skills:
        for skill in agent_card.skills:
            logger.info(f"  Skill: {skill.name} (ID: {skill.id}, Tags: {skill.tags})")

    # Run the Uvicorn server
    uvicorn.run(a2a_app.build(), host=host, port=port)


if __name__ == "__main__":
    main()
