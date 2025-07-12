"""ElevenLabs Agent implementation using ADK and MCPToolset with custom timeout patch."""

import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from mcp import StdioServerParameters

from config import ELEVENLABS_API_KEY, GOOGLE_API_KEY
from elevenlabs_agent.prompt import ELEVENLABS_PROMPT
from utils.custom_adk_patches import CustomMCPToolset


def create_elevenlabs_agent() -> Agent:
    return Agent(
        name="elevenlabs_agent_mcp",
        model=LiteLlm(
            model="gemini/gemini-2.0-flash", 
            api_key=GOOGLE_API_KEY
        ),  
        description="Specialized agent for converting text to speech using ElevenLabs via MCPToolset.",
        instruction=ELEVENLABS_PROMPT,
        tools=[
            CustomMCPToolset(
                connection_params=StdioServerParameters(
                    command='uvx',
                    args=['elevenlabs-mcp'], 
                    env={"ELEVENLABS_API_KEY": ELEVENLABS_API_KEY}
                )
            )
        ]
    ) 


root_agent = create_elevenlabs_agent()