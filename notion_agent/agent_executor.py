"""Notion Agent Executor for A2A integration."""

import logging
import uuid
import datetime
from collections.abc import AsyncGenerator

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session as ADKSession
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as adk_types

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Part, TextPart, AgentCard
from a2a.utils import new_agent_text_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app name for the runner, specific to this agent
NOTION_A2A_APP_NAME = "notion_a2a_app"


class NotionADKAgentExecutor(AgentExecutor):
    """ADK Agent Executor for Notion A2A integration."""
    
    def __init__(self, agent: Agent, agent_card: AgentCard):
        """Initialize with an Agent instance and set up ADK Runner.
        
        Args:
            agent: The Notion ADK agent instance
            agent_card: Agent card for A2A service registration
        """
        logger.info(f"Initializing NotionADKAgentExecutor for agent: {agent.name}")
        self.agent = agent
        self._card = agent_card
        
        # Initialize ADK services
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=NOTION_A2A_APP_NAME,
            session_service=self.session_service,
            artifact_service=self.artifact_service
        )
        logger.info(f"ADK Runner initialized for app '{self.runner.app_name}' for agent '{self.agent.name}'")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the Notion agent's logic for a given request context.
        
        Args:
            context: The A2A request context containing user input
            event_queue: Queue for sending events back to the A2A client
        """
        try:
            user_input = context.get_user_input()
            if not user_input:
                logger.warning(f"No user input found for {self.agent.name}; using default search.")
                user_input = "Search for recent pages"
            
            logger.info(f"{self.agent.name} processing search query: '{user_input}'")
            
            # Use a consistent user_id for Notion searches
            user_id = "a2a_user_notion"
            # Use task_id as session_id if available, otherwise generate new
            session_id = context.task_id or str(uuid.uuid4())
            
            # Create or get ADK session
            adk_session: ADKSession | None = await self.session_service.get_session(
                app_name=self.runner.app_name, 
                user_id=user_id, 
                session_id=session_id
            )
            if not adk_session:
                adk_session = await self.session_service.create_session(
                    app_name=self.runner.app_name, 
                    user_id=user_id, 
                    session_id=session_id, 
                    state={}
                )
                logger.info(f"Created new ADK session: {session_id} for {self.agent.name}")
            
            request_content = adk_types.Content(
                role="user", 
                parts=[adk_types.Part(text=user_input)]
            )
            
            logger.debug(f"Running ADK agent {self.agent.name} with session {session_id}")
            events_async: AsyncGenerator[adk_types.Event, None] = self.runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=request_content
            )
            
            final_message_text = "(No search results found)"
            
            async for event in events_async:
                if event.is_final_response() and event.content and event.content.role == "model":
                    if event.content.parts and event.content.parts[0].text:
                        final_message_text = event.content.parts[0].text
                        logger.info(f"{self.agent.name} final response: '{final_message_text[:200]}{'...' if len(final_message_text) > 200 else ''}'")
                        break
                    else:
                        logger.warning(f"{self.agent.name} received final event but no text in first part: {event.content.parts}")
                elif event.is_final_response():
                     logger.warning(f"{self.agent.name} received final event without model content: {event}")

            logger.info(f"Sending Notion search response for task {context.task_id}")
            event_queue.enqueue_event(
                new_agent_text_message(
                    text=final_message_text, 
                    context_id=context.context_id, 
                    task_id=context.task_id
                )
            )

        except Exception as e:
            logger.error(f"Error executing Notion search in {self.agent.name}: {str(e)}", exc_info=True)
            error_message_text = f"Error searching Notion workspace: {str(e)}"
            event_queue.enqueue_event(
                new_agent_text_message(
                    text=error_message_text,
                    context_id=context.context_id,
                    task_id=context.task_id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Request the agent to cancel an ongoing task.
        
        Args:
            context: The A2A request context
            event_queue: Queue for sending cancellation events
        """
        task_id = context.task_id or "unknown_task"
        context_id = context.context_id or "unknown_context"
        logger.info(f"Cancelling Notion search task: {task_id} for agent {self.agent.name}")
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        try:
            state_cancelled = TaskState.canceled 
        except AttributeError:
            logger.warning("TaskState.canceled not found, using string 'cancelled'.")
            state_cancelled = "cancelled"

        canceled_status = TaskStatus(
            state=state_cancelled,
            timestamp=timestamp 
        )
        cancel_event = TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=canceled_status,
            final=True
        )
        event_queue.enqueue_event(cancel_event)
        logger.info(f"Sent cancel event for Notion task: {task_id}") 