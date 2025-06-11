"""Notion Agent A2A Service Entry Point."""

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

from notion_agent.agent import create_notion_agent

# Local agent imports
from notion_agent.agent_executor import NotionADKAgentExecutor

# Load environment variables from .env file
load_dotenv()

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    "host",
    default=os.getenv("A2A_NOTION_HOST", "localhost"),
    show_default=True,
    help="Host for the Notion agent server.",
)
@click.option(
    "--port",
    "port",
    default=int(os.getenv("A2A_NOTION_PORT", 8002)),
    show_default=True,
    type=int,
    help="Port for the Notion agent server.",
)
def main(host: str, port: int) -> None:
    """Runs the Notion ADK Agent as an A2A service."""

    # Check for required API key
    if not os.getenv("NOTION_API_KEY"):
        logger.warning(
            "NOTION_API_KEY environment variable not set. "
            "The Notion MCP server might fail to authenticate."
        )

    # Define AgentCard for Notion
    notion_skill = AgentSkill(
        id="notion_search",
        name="Search Notion workspace",
        description="Searches and retrieves information from Notion pages, databases, and blocks.",
        tags=["notion", "search", "retrieval", "knowledge", "workspace"],
        examples=[
            "Search for 'project plan'",
            "Find pages about Q3 goals",
            "Retrieve information from the meeting notes database",
        ],
    )

    agent_card = AgentCard(
        name="Notion Search Agent",
        description="Provides information retrieval services from Notion workspace using MCP.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[notion_skill],
    )

    try:
        # Create the actual ADK Agent
        adk_agent = create_notion_agent()

        # Initialize the ADK Runner (following official ADK pattern)
        runner = Runner(
            app_name=agent_card.name,
            agent=adk_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        # Instantiate the AgentExecutor with the runner
        agent_executor = NotionADKAgentExecutor(
            agent=adk_agent, agent_card=agent_card, runner=runner
        )

    except Exception as e:
        logger.error(
            f"Failed to initialize Notion Agent components: {e}", exc_info=True
        )
        return

    # Set up the A2A request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    # Create the A2A Starlette application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info(f"Starting Notion Agent server on http://{host}:{port}")
    logger.info(f"Agent Name: {agent_card.name}, Version: {agent_card.version}")
    if agent_card.skills:
        for skill in agent_card.skills:
            logger.info(f"  Skill: {skill.name} (ID: {skill.id}, Tags: {skill.tags})")

    # Run the Uvicorn server
    uvicorn.run(a2a_app.build(), host=host, port=port)


if __name__ == "__main__":
    main()
