import logging
import uuid
import datetime
# import asyncio # No longer explicitly needed for this simpler structure's async for
from collections.abc import AsyncGenerator # Still needed for _run_agent type hint if kept

from google.adk.agents import Agent # Changed from LlmAgent to base Agent if preferred by simpler model
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session as ADKSession
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Added
from google.genai import types as adk_types # Renamed from genai_types for consistency with simpler example

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Part, TextPart, AgentCard # TaskState might need "string" access
from a2a.utils import new_agent_text_message # Used for sending response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app name for the runner, specific to this agent
ELEVENLABS_A2A_APP_NAME = "elevenlabs_a2a_app"

class ElevenLabsADKAgentExecutor(AgentExecutor):
    """ADK Agent Executor for A2A integration - Simplified to match speaker_agent example."""
    
    def __init__(self, agent: Agent, agent_card: AgentCard): # agent_card is still needed by __main__.py for A2AStarletteApplication
        """Initialize with an Agent instance and set up ADK Runner."""
        logger.info(f"Initializing ElevenLabsADKAgentExecutor for agent: {agent.name}")
        self.agent = agent
        self._card = agent_card # Store agent_card
        
        # Initialize ADK services
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=ELEVENLABS_A2A_APP_NAME, # Use specific app name
            session_service=self.session_service,
            artifact_service=self.artifact_service # Add artifact service
            # memory_service can be added if the agent uses it.
        )
        logger.info(f"ADK Runner initialized for app '{self.runner.app_name}' for agent '{self.agent.name}'")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the agent's logic for a given request context.
        Simplified to mirror the speaker_agent example.
        """
        try:
            user_input = context.get_user_input()
            if not user_input:
                logger.warning(f"No user input found for {self.agent.name}; using default 'Hello'.")
                user_input = "Hello" # Default if no input
            
            logger.info(f"{self.agent.name} processing: '{user_input}'")
            
            # Use a consistent user_id for simplicity in this model
            user_id = "a2a_user_elevenlabs" 
            # Use task_id as session_id if available, otherwise generate new
            session_id = context.task_id or str(uuid.uuid4())
            
            # Create or get session (ADK Session, not A2A Task)
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
                    state={} # Initial empty state for ADK session
                )
                logger.info(f"Created new ADK session: {session_id} for {self.agent.name}")
            
            request_content = adk_types.Content(
                role="user", 
                parts=[adk_types.Part(text=user_input)]
            )
            
            logger.debug(f"Running ADK agent {self.agent.name} with session {session_id}")
            events_async: AsyncGenerator[adk_types.Event, None] = self.runner.run_async( # ADK Event
                user_id=user_id, 
                session_id=session_id, 
                new_message=request_content
            )
            
            final_message_text = "(No response generated)"
            
            async for event in events_async:
                if event.is_final_response() and event.content and event.content.role == "model":
                    if event.content.parts and event.content.parts[0].text:
                        final_message_text = event.content.parts[0].text
                        logger.info(f"{self.agent.name} final response: '{final_message_text}'")
                        break
                    else:
                        logger.warning(f"{self.agent.name} received final event but no text in first part: {event.content.parts}")
                elif event.is_final_response():
                     logger.warning(f"{self.agent.name} received final event without model content: {event}")


            logger.info(f"Sending response for {self.agent.name} task {context.task_id}: '{final_message_text}'")
            # This sends a simple text message back. 
            # It doesn't create formal A2A artifacts or use TaskUpdater.
            event_queue.enqueue_event(
                new_agent_text_message(
                    text=final_message_text, 
                    context_id=context.context_id, 
                    task_id=context.task_id
                )
            )
            # To be more A2A compliant for task completion, one might also send a TaskStatusUpdateEvent.
            # For simplicity here, mirroring the example, only the text message is sent.
            # If the simple example also updated task status, that should be added here.
            # Example:
            # completion_status = TaskStatus(state=TaskState("completed"), timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
            # event_queue.enqueue_event(TaskStatusUpdateEvent(taskId=context.task_id, contextId=context.context_id, status=completion_status, final=True))

        except Exception as e:
            logger.error(f"Error executing {self.agent.name}: {str(e)}", exc_info=True)
            error_message_text = f"Error processing your request for {self.agent.name}: {str(e)}"
            event_queue.enqueue_event(
                new_agent_text_message(
                    text=error_message_text,
                    context_id=context.context_id,
                    task_id=context.task_id
                )
            )
            # Also send a failed status
            # failed_status = TaskStatus(state=TaskState("failed"), timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), message=Part(root=TextPart(text=str(e)))) # Create Part for message
            # event_queue.enqueue_event(TaskStatusUpdateEvent(taskId=context.task_id, contextId=context.context_id, status=failed_status, final=True))


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Request the agent to cancel an ongoing task.
        Mirrors the speaker_agent example.
        """
        task_id = context.task_id or "unknown_task"
        context_id = context.context_id or "unknown_context"
        logger.info(f"Cancelling task: {task_id} for agent {self.agent.name}")
        
        # Ensure datetime is timezone-aware (UTC)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Use string for TaskState if direct attribute access is problematic
        # This matches the simpler example's direct use, assuming TaskState can be string.
        try:
            state_cancelled = TaskState.canceled 
        except AttributeError:
            logger.warning("TaskState.canceled not found, using string 'cancelled'. Ensure a2a.types.TaskState supports this.")
            state_cancelled = "cancelled"

        canceled_status = TaskStatus(
            state=state_cancelled, # Use string or direct attribute
            timestamp=timestamp 
            # message=Part(root=TextPart(text="Task cancelled by A2A request.")) # Optional message
        )
        cancel_event = TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            # kind="status-update", # 'kind' is often not needed if type is clear by class
            status=canceled_status,
            final=True # Cancellation is a final state
        )
        event_queue.enqueue_event(cancel_event)
        logger.info(f"Sent cancel event for task: {task_id}")

# Helper functions (convert_a2a_parts_to_genai, convert_genai_parts_to_a2a) are removed
# as this simplified executor does not use them. It directly handles text.