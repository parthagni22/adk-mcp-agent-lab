# notion_agent/test_client.py
import asyncio
import os
import traceback
from typing import Any, Dict
from uuid import uuid4

import httpx
from a2a.client import A2AClient
from a2a.types import (
    GetTaskRequest,
    GetTaskResponse,
    GetTaskSuccessResponse,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskQueryParams,
)
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

AGENT_URL = os.getenv("NOTION_AGENT_A2A_URL", "http://localhost:8002")


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> Dict[str, Any]:
    """Create a message payload for A2A communication.

    Args:
        text: The message text to send to the agent
        task_id: Optional task ID for continuing a conversation
        context_id: Optional context ID for conversation context

    Returns:
        Dictionary containing the message payload
    """
    payload: Dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"text": text}],
            "messageId": uuid4().hex,
        },
    }
    if task_id:
        payload["message"]["taskId"] = task_id
    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


def print_json_response(response: Any, description: str) -> None:
    """Print a JSON response with a description for debugging.

    Args:
        response: The response object to print
        description: A description of what this response represents
    """
    print(f"--- {description} ---")
    try:
        if hasattr(response, "model_dump_json"):
            print(response.model_dump_json(exclude_none=True))
        elif hasattr(response, "root") and hasattr(response.root, "model_dump_json"):
            print(response.root.model_dump_json(exclude_none=True))
        elif hasattr(response, "dict"):
            print(response.dict(exclude_none=True))
        else:
            print(str(response))
        print()  # Add a newline after the JSON
    except Exception as e:
        print(f"Error printing response: {e}")
        print(str(response))
        print()


async def _execute_and_poll_task(client: A2AClient, query: str, test_name: str) -> None:
    """Helper function to send a query to the agent and poll for the result."""
    print(f"\n--- Starting Test: {test_name} ---")
    print(f"Query: {query}")

    send_payload = create_send_message_payload(text=query)
    request = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_payload)
    )

    print("\n--- Sending Task ---")
    send_response: SendMessageResponse = await client.send_message(request)
    print_json_response(send_response, f"Send Task Response ({test_name})")

    if (
        not isinstance(send_response, SendMessageSuccessResponse)
        or not send_response.result
    ):
        print("Received non-success or empty result from send_message. Aborting.")
        return

    agent_reply_data = send_response.result

    extracted_task_id: str | None = None
    if hasattr(agent_reply_data, "taskId"):
        task_id_value = getattr(agent_reply_data, "taskId", None)
        if isinstance(task_id_value, str):
            extracted_task_id = task_id_value

    if not extracted_task_id and isinstance(agent_reply_data, dict):
        task_id_value = agent_reply_data.get("taskId")
        if isinstance(task_id_value, str):
            extracted_task_id = task_id_value

    if not extracted_task_id:
        print("Could not extract taskId from the agent's reply. Aborting.")
        print_json_response(
            agent_reply_data, "Agent's reply (send_response.result) for debugging"
        )
        return

    task_id: str = extracted_task_id
    print(f"Task ID (from agent reply): {task_id}")
    print("\n--- Querying Task Status ---")

    max_retries = 10
    retry_delay = 2  # seconds
    task_status = "unknown"

    for attempt in range(max_retries):
        get_request = GetTaskRequest(
            id=str(uuid4()), params=TaskQueryParams(id=task_id)
        )
        get_response: GetTaskResponse = await client.get_task(get_request)
        print_json_response(get_response, f"Get Task Response (Attempt {attempt + 1})")

        if isinstance(get_response, GetTaskSuccessResponse) and get_response.result:
            actual_task_result = get_response.result
            if actual_task_result.status:
                task_status = actual_task_result.status.state
                print(f"Task State: {task_status}")
                if task_status in ["completed", "failed"]:
                    if task_status == "completed" and actual_task_result.artifacts:
                        print("\n--- Final Artifacts ---")
                        for i, artifact_item in enumerate(actual_task_result.artifacts):
                            if isinstance(artifact_item, dict):
                                parts_list = artifact_item.get("parts")
                                if isinstance(parts_list, list):
                                    for j, part_data in enumerate(parts_list):
                                        if isinstance(part_data, dict):
                                            print(f"  Artifact {i}, Part {j}:")
                                            for key, value in part_data.items():
                                                print(f"    {key}: {value}")
                                        else:
                                            print(
                                                f"  Artifact {i}, Part {j} (unexpected item type): {part_data}"
                                            )
                                else:
                                    print(
                                        f"  Artifact {i} (no 'parts' list or not a list): {artifact_item}"
                                    )
                            else:
                                print(
                                    f"  Artifact {i} (unexpected type, not a dict): {artifact_item}"
                                )
                        print("✅ Test complete.")
                    elif task_status == "failed" and actual_task_result.status.message:
                        print(f"❌ Task Failed: {actual_task_result.status.message}")
                    break
            else:
                print("GetTaskResponse result did not contain status.")
        else:
            print(
                "GetTaskResponse was not successful or did not contain expected result structure."
            )

        if attempt < max_retries - 1 and task_status not in ["completed", "failed"]:
            print(f"Task not final, retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        elif task_status in ["completed", "failed"]:
            break
        else:
            print("Max retries reached, task did not complete.")
            break


async def run_notion_search_test(client: A2AClient) -> None:
    """Test the Notion agent's search functionality using the 'sermon notes' use case."""
    await _execute_and_poll_task(
        client,
        "Search my Notion workspace for pages related to 'sermon notes'.",
        "Notion Search Test",
    )


async def run_notion_database_test(client: A2AClient) -> None:
    """Test the Notion agent's database query functionality, specifically counting."""
    await _execute_and_poll_task(
        client,
        "Count the total number of entries in the 'Sermon Notes' database and return only the number.",
        "Notion Database Count Test",
    )


async def main() -> None:
    """Main function to run Notion agent tests."""
    print(f"Connecting to Notion Agent at {AGENT_URL}...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, AGENT_URL
            )
            print("Connection successful.")

            # Run tests sequentially
            await run_notion_search_test(client)
            await run_notion_database_test(client)

    except httpx.ConnectError as e:
        print(
            f"\n❌ Connection error: Could not connect to agent at {AGENT_URL}. Ensure the server is running."
        )
        print(f"Details: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
