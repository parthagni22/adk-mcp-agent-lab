"""Test client for Host Agent A2A functionality."""

import asyncio
import httpx
import os
import traceback
from uuid import uuid4
from typing import Any, Dict

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

# Load environment variables from .env
load_dotenv()

AGENT_URL = os.getenv("HOST_AGENT_A2A_URL", "http://localhost:8001")


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


def print_json_response(response: Any, description: str) -> None:
    """Print JSON response in a readable format.
    
    Args:
        response: The response object to print
        description: Description of the response
    """
    print(f"--- {description} ---")
    try:
        if hasattr(response, 'model_dump_json'):
            print(response.model_dump_json(exclude_none=True))
        elif hasattr(response, 'root') and hasattr(response.root, 'model_dump_json'):
            print(response.root.model_dump_json(exclude_none=True))
        elif hasattr(response, 'dict'): 
            print(response.dict(exclude_none=True))
        else:
            print(str(response))
        print()  # Add a newline after the JSON
    except Exception as e:
        print(f"Error printing response: {e}")
        print(str(response))
        print()


async def run_host_agent_test(client: A2AClient, test_query: str, test_name: str) -> None:
    """Run a test query against the host agent.
    
    Args:
        client: The A2A client to use
        test_query: The test query to send
        test_name: Name of the test for logging
    """
    print(f"\n=== {test_name} ===")
    print(f"Test Query: {test_query}")

    send_payload = create_send_message_payload(text=test_query)
    request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_payload))

    print("\n--- Sending Task ---")
    send_response: SendMessageResponse = await client.send_message(request)
    print_json_response(send_response, 'Send Task Response')

    if not isinstance(send_response, SendMessageSuccessResponse) or not send_response.result:
        print('Received non-success or empty result from send_message. Aborting.')
        return

    # Extract task ID from agent's reply
    agent_reply_data = send_response.result
    extracted_task_id: str | None = None

    # Handle both Pydantic models and dict responses
    if hasattr(agent_reply_data, 'taskId'):
        task_id_value = getattr(agent_reply_data, 'taskId', None)
        if isinstance(task_id_value, str):
            extracted_task_id = task_id_value
    
    if not extracted_task_id and isinstance(agent_reply_data, dict):
        task_id_value = agent_reply_data.get('taskId')
        if isinstance(task_id_value, str):
            extracted_task_id = task_id_value

    if not extracted_task_id:
        print("Could not extract taskId from the agent's reply. Aborting.")
        print_json_response(agent_reply_data, "Agent's reply for debugging")
        return

    task_id: str = extracted_task_id
    print(f"Task ID (from agent reply): {task_id}")
    print("\n--- Querying Task Status ---")
    
    max_retries = 15  # Longer timeout for orchestration workflows
    retry_delay = 3   # Longer delay for multi-agent calls
    task_status = "unknown"

    for attempt in range(max_retries):
        get_request = GetTaskRequest(id=str(uuid4()), params=TaskQueryParams(id=task_id))
        get_response: GetTaskResponse = await client.get_task(get_request)
        print_json_response(get_response, f'Get Task Response (Attempt {attempt + 1})')

        if isinstance(get_response, GetTaskSuccessResponse) and get_response.result:
            actual_task_result = get_response.result 
            if actual_task_result.status:
                task_status = actual_task_result.status.state
                print(f"Task State: {task_status}")
                
                if task_status in ["completed", "failed"]:
                    if task_status == "completed" and actual_task_result.artifacts:
                        print("\n--- Final Results ---")
                        for i, artifact_item in enumerate(actual_task_result.artifacts):
                            if isinstance(artifact_item, dict):
                                parts_list = artifact_item.get('parts')
                                if isinstance(parts_list, list):
                                    for j, part_data in enumerate(parts_list):
                                        if isinstance(part_data, dict):
                                            print(f"  Result {i}, Part {j}:")
                                            text = part_data.get('text')
                                            audio_url = part_data.get('audio_url')
                                            if text:
                                                print(f"    Text: {text}")
                                            if audio_url:
                                                print(f"    Audio URL: {audio_url}")
                                        else:
                                            print(f"  Result {i}, Part {j}: {part_data}")
                                else:
                                    print(f"  Result {i}: {artifact_item}")
                            else:
                                print(f"  Result {i}: {artifact_item}")
                        print("✅ Test complete.")
                    elif task_status == "failed" and actual_task_result.status.message:
                        print(f"❌ Task Failed: {actual_task_result.status.message}")
                    break
            else:
                print("GetTaskResponse result did not contain status.")
        else:
            print("GetTaskResponse was not successful or did not contain expected result structure.")
        
        if attempt < max_retries - 1 and task_status not in ["completed", "failed"]:
            print(f"Task not final, retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        elif task_status in ["completed", "failed"]:
            break
        else:
            print("Max retries reached, task did not complete.")
            break


async def run_all_tests(client: A2AClient) -> None:
    """Run all host agent tests based on the 'sermon notes' use case."""
    # Test 1: Simple Notion query about sermon notes
    await run_host_agent_test(
        client,
        "Search my Notion workspace for 'sermon notes'",
        "Notion Agent Test (Sermon Notes Search)"
    )
    
    # Test 2: Database query orchestration
    await run_host_agent_test(
        client,
        "Count how many sermon notes are in the database.",
        "Database Query Orchestration Test"
    )

    # Test 3: Full orchestration workflow for sermon notes
    await run_host_agent_test(
        client,
        "Find my sermon notes and read the titles of the first two entries aloud.",
        "Full Orchestration Test (Sermon Notes)"
    )
    
    # Test 4: Simple ElevenLabs TTS (unrelated to sermons, good for isolation)
    await run_host_agent_test(
        client,
        "Convert this text to speech: 'This is a test of the text to speech system.'",
        "ElevenLabs Agent Isolation Test"
    )


async def main() -> None:
    """Main test function."""
    print(f'Connecting to Host Agent at {AGENT_URL}...')
    try:
        async with httpx.AsyncClient(timeout=90.0) as httpx_client:  # Longer timeout for orchestration
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, AGENT_URL
            )
            print('Connection successful.')
            await run_all_tests(client)

    except httpx.ConnectError as e:
        print(f'\n❌ Connection error: Could not connect to agent at {AGENT_URL}. Ensure the server is running.')
        print('To start the full system (child agents and host agent):')
        print('python scripts/start_agents.py')
        print(f'Details: {e}')
        traceback.print_exc()
    except Exception as e:
        print(f'\n❌ An unexpected error occurred: {e}')
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 