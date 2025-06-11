"""Remote connections for A2A client communication following the architecture pattern."""

import asyncio
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
    TaskQueryParams,
)


class RemoteConnections:
    """Handles A2A connections and communication with downstream agents.

    Follows the Remote Connections Pattern from @adk-a2a.mdc:
    - Send A2A message via client
    - Poll for task completion (or handle immediate responses)
    - Extract text artifacts from completed tasks
    - Return result or error dict
    """

    def __init__(self, httpx_client: httpx.AsyncClient):
        """Initialize RemoteConnections with httpx client.

        Args:
            httpx_client: The async HTTP client to use for A2A communication
        """
        self.httpx_client = httpx_client

    @classmethod
    async def create(cls, timeout: float = 30.0) -> "RemoteConnections":
        """Factory method to create RemoteConnections with managed httpx client.

        Args:
            timeout: HTTP timeout in seconds

        Returns:
            RemoteConnections: Configured instance
        """
        httpx_client = httpx.AsyncClient(timeout=timeout)
        return cls(httpx_client)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self.httpx_client.aclose()

    async def invoke_agent(self, agent_url: str, query: str) -> Dict[str, Any]:
        """Invoke a downstream agent via A2A protocol.

        Args:
            agent_url: The URL of the target A2A agent
            query: The query/message to send to the agent

        Returns:
            Dict with either:
            - {"result": "success text response"} for success
            - {"error": "error message"} for failures
        """
        try:
            # Create A2A client from agent card
            client = await A2AClient.get_client_from_agent_card_url(
                self.httpx_client, agent_url
            )

            # Send A2A message
            send_payload = self._create_send_message_payload(query)
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_payload)
            )

            send_response: SendMessageResponse = await client.send_message(request)

            # Handle wrapped response structure
            if hasattr(send_response, "root") and isinstance(
                send_response.root, SendMessageSuccessResponse
            ):
                success_response = send_response.root
            elif isinstance(send_response, SendMessageSuccessResponse):
                success_response = send_response
            else:
                return {"error": f"Failed to send message to agent at {agent_url}"}

            if not success_response.result:
                return {"error": f"No result from agent at {agent_url}"}

            # Try to extract immediate response first
            immediate_result = self._extract_immediate_response(success_response.result)
            if immediate_result:
                return {"result": immediate_result}

            # If no immediate response, try task-based polling
            task_id = self._extract_task_id(success_response.result)
            if task_id:
                return await self._poll_task_completion(client, task_id)

            return {
                "error": "No immediate response content and could not extract taskId for polling"
            }

        except httpx.ConnectError:
            return {
                "error": f"Could not connect to agent at {agent_url}. Ensure the server is running."
            }
        except Exception as e:
            return {"error": f"Error calling agent at {agent_url}: {str(e)}"}

    def _create_send_message_payload(
        self, text: str, task_id: str | None = None, context_id: str | None = None
    ) -> Dict[str, Any]:
        """Create A2A send message payload with proper format.

        Args:
            text: The message text to send
            task_id: Optional task ID to associate with the message
            context_id: Optional context ID for conversation context

        Returns:
            dict: Properly formatted A2A message payload
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

    def _extract_immediate_response(self, agent_reply_data: Any) -> str | None:
        """Extract immediate response content from agent reply.

        Args:
            agent_reply_data: The reply data from the agent

        Returns:
            str: Extracted text content, or None if not found
        """
        try:
            # Check for immediate response content (many A2A agents return results directly)
            if hasattr(agent_reply_data, "parts") and agent_reply_data.parts:
                immediate_results = []
                for part in agent_reply_data.parts:
                    # Try to extract text from part structure
                    text_content = None

                    # Check for part.root.text (most common pattern)
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_content = getattr(part.root, "text", None)
                    # Fallback to direct part.text
                    elif hasattr(part, "text"):
                        text_content = getattr(part, "text", None)

                    if text_content and isinstance(text_content, str):
                        immediate_results.append(text_content)

                if immediate_results:
                    return "\n".join(immediate_results)

            # Check if the response itself has text content
            if hasattr(agent_reply_data, "text") and agent_reply_data.text:
                return str(agent_reply_data.text)

        except (AttributeError, TypeError):
            # If immediate parsing fails, return None to try task polling
            pass

        return None

    def _extract_task_id(self, agent_reply_data: Any) -> str | None:
        """Extract task ID from agent reply for polling.

        Args:
            agent_reply_data: The reply data from the agent

        Returns:
            str: Task ID if found, None otherwise
        """
        # Handle both Pydantic models and dict responses for task ID
        if hasattr(agent_reply_data, "taskId"):
            task_id_value = getattr(agent_reply_data, "taskId", None)
            if isinstance(task_id_value, str):
                return task_id_value
        elif isinstance(agent_reply_data, dict):
            task_id_value = agent_reply_data.get("taskId")
            if isinstance(task_id_value, str):
                return task_id_value

        return None

    async def _poll_task_completion(
        self, client: A2AClient, task_id: str
    ) -> Dict[str, Any]:
        """Poll for task completion and extract results.

        Args:
            client: The A2A client to use for polling
            task_id: The task ID to poll

        Returns:
            Dict with result or error
        """
        max_retries = 10
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                get_request = GetTaskRequest(
                    id=str(uuid4()), params=TaskQueryParams(id=task_id)
                )
                get_response: GetTaskResponse = await client.get_task(get_request)

                if (
                    isinstance(get_response, GetTaskSuccessResponse)
                    and get_response.result
                ):
                    actual_task_result = get_response.result
                    if actual_task_result.status:
                        task_status = actual_task_result.status.state

                        if task_status == "completed":
                            # Extract artifacts
                            if actual_task_result.artifacts:
                                # Collect all text and audio URLs from artifacts
                                results = []
                                for artifact_item in actual_task_result.artifacts:
                                    if isinstance(artifact_item, dict):
                                        parts_list = artifact_item.get("parts")
                                        if isinstance(parts_list, list):
                                            for part_data in parts_list:
                                                if isinstance(part_data, dict):
                                                    text = part_data.get("text")
                                                    audio_url = part_data.get(
                                                        "audio_url"
                                                    )
                                                    if text:
                                                        results.append(text)
                                                    if audio_url:
                                                        results.append(
                                                            f"Audio URL: {audio_url}"
                                                        )

                                if results:
                                    return {"result": "\n".join(results)}
                                else:
                                    return {"result": "No content received"}

                        elif task_status == "failed":
                            error_msg = (
                                actual_task_result.status.message
                                if actual_task_result.status.message
                                else "Task failed"
                            )
                            return {"error": error_msg}

                        elif task_status in ["pending", "running"]:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return {"error": "Task did not complete within timeout"}

                # If we get here, the response wasn't as expected
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {"error": "Task polling failed - unexpected response format"}

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return {"error": f"Task polling error: {str(e)}"}

        return {"error": "Task did not complete within timeout"}
