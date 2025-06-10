"""ElevenLabs Agent implementation using ADK and MCPToolset with custom timeout patch."""

import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from mcp import StdioServerParameters

from config import ADK_MODEL, ELEVENLABS_API_KEY, GOOGLE_API_KEY
from elevenlabs_agent.prompt import ELEVENLABS_PROMPT
from utils.custom_adk_patches import CustomMCPToolset as MCPToolset

def create_elevenlabs_agent() -> Agent:
    """Creates an ADK agent specialized for text-to-speech via ElevenLabs MCP.
    
    The agent is configured to launch the ElevenLabs MCP server using npx.
    The ELEVENLABS_API_KEY environment variable must be available for the MCP server.
    
    Returns:
        LlmAgent: ADK LlmAgent configured with MCPToolset for ElevenLabs.
    """
    return Agent(
        name="elevenlabs_agent_mcp",
        model=LiteLlm(model="anthropic/claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY")),  # Switch to Claude 3.5
        description="Specialized agent for converting text to speech using ElevenLabs via MCPToolset.",
        instruction=ELEVENLABS_PROMPT,
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command='uvx',
                    args=['elevenlabs-mcp'], 
                    env={"ELEVENLABS_API_KEY": ELEVENLABS_API_KEY}
                )
            )
        ]
    ) 

root_agent = create_elevenlabs_agent()