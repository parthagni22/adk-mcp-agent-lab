"""Notion Agent implementation using ADK and MCPToolset."""

import json
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

from config import ADK_MODEL, NOTION_API_KEY, GOOGLE_API_KEY
from notion_agent.prompt import NOTION_PROMPT


def create_notion_agent() -> Agent:
    """Creates an ADK agent specialized for Notion information retrieval via MCP.
    
    The agent is configured to launch the Notion MCP server using npx.
    The NOTION_API_KEY environment variable must be available for the MCP server.
    
    Based on google-adk documentation, this version relies on ADK's built-in 
    error handling to manage any schema validation issues gracefully.
    
    Returns:
        Agent: ADK Agent configured with MCPToolset for Notion.
    """
    return Agent(
        name="notion_agent_mcp",
        model="gemini-2.0-flash",
        description="Specialized agent for retrieving information from Notion workspace via MCPToolset.",
        instruction=NOTION_PROMPT,
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=['-y', '@notionhq/notion-mcp-server'], 
                    env={"OPENAPI_MCP_HEADERS": json.dumps({
                        "Authorization": f"Bearer {NOTION_API_KEY}",
                        "Notion-Version": "2022-06-28"
                    })}
                )
                # No tool_filter - let ADK handle all available tools
            )
        ]
    )


root_agent = create_notion_agent() 