#!/usr/bin/env python3
"""Multi-Agent Startup Script.

This script provides a convenient way to start, manage, and monitor the
A2A (Agent-to-Agent) services required for this project. It handles
the startup of child agents and the host agent, checks for required
API keys, and ensures graceful shutdown.

USAGE:
- Start all agents (default behavior):
  $ python start_agents.py

- Start only child agents (Notion, ElevenLabs):
  $ python start_agents.py --no-with-host

FEATURES:
- Uses `click` for a clean command-line interface.
- Loads environment variables from `.env`.
- Checks for required API keys before starting.
- Starts each agent in a separate subprocess.
- Monitors agent health by checking their agent card endpoints.
- Logs all agent output to files in the logs/ directory.
- Gracefully terminates all child processes on exit (Ctrl+C).
"""
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Agent Configurations ---
AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "notion": {
        "module": "notion_agent",
        "port": 8002,
        "host": "localhost",
        "env_var": "NOTION_API_KEY",
    },
    "elevenlabs": {
        "module": "elevenlabs_agent",
        "port": 8003,
        "host": "localhost",
        "env_var": "ELEVENLABS_API_KEY",
    },
    "host": {
        "module": "host_agent",
        "port": 8001,
        "host": "localhost",
        "env_var": None,  # No direct API key needed
    },
}


def check_env_vars(agents_to_start: List[str]) -> None:
    """Check for required environment variables before starting agents.
    
    Args:
        agents_to_start: A list of agent names to check.
        
    Raises:
        click.UsageError: If a required environment variable is not set.
    """
    for agent_name in agents_to_start:
        config = AGENT_CONFIGS[agent_name]
        env_var = config.get("env_var")
        if env_var and not os.getenv(env_var):
            raise click.UsageError(
                f"Missing environment variable: {env_var} is required for '{agent_name}' agent. "
                "Please set it in your .env file."
            )
    print("✅ All required environment variables are set.")


async def wait_for_service(agent_name: str, url: str, timeout: int = 30) -> bool:
    """Wait for an agent's A2A service to be ready.
    
    Args:
        agent_name: The name of the agent.
        url: The URL of the agent's agent card.
        timeout: The maximum time to wait in seconds.
        
    Returns:
        True if the service becomes ready, False otherwise.
    """
    start_time = time.time()
    print(f"⏳ Waiting for {agent_name} agent to be ready at {url}...")
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ {agent_name.title()} agent is ready!")
                    return True
        except (httpx.ConnectError, httpx.TimeoutException):
            await asyncio.sleep(1)  # Wait before retrying
    
    print(f"❌ {agent_name.title()} agent failed to start within {timeout}s.")
    return False


def ensure_logs_directory() -> Path:
    """Ensure the logs directory exists and return its path.
    
    Returns:
        Path: The path to the logs directory.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


@click.command()
@click.option(
    "--with-host/--no-with-host",
    default=True,
    help="Start the host agent in addition to child agents. Default is --with-host.",
)
def main(with_host: bool) -> None:
    """Starts and manages the A2A agent services."""
    
    agents_to_start = ["notion", "elevenlabs"]
    if with_host:
        agents_to_start.append("host")

    # Ensure logs directory exists
    logs_dir = ensure_logs_directory()
    print(f"📁 Agent logs will be saved to: {logs_dir.absolute()}")

    try:
        check_env_vars(agents_to_start)
    except click.UsageError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    processes: Dict[str, subprocess.Popen] = {}
    
    try:
        # Start all specified agents
        for agent_name in agents_to_start:
            config = AGENT_CONFIGS[agent_name]
            cmd = [
                "python", "-m", config["module"],
                "--host", config["host"],
                "--port", str(config["port"])
            ]
            
            print(f"🚀 Starting {agent_name} agent with command: {' '.join(cmd)}")
            
            # Create log files for each agent
            stdout_log = logs_dir / f"{agent_name}_agent_stdout.log"
            stderr_log = logs_dir / f"{agent_name}_agent_stderr.log"
            
            print(f"   📋 Logs: {stdout_log} & {stderr_log}")
            
            # Use Popen to start the process without blocking, redirect output to log files
            with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True
                )
                processes[agent_name] = process

        # Asynchronously wait for all services to be ready
        async def wait_for_all_services():
            tasks = []
            for agent_name in agents_to_start:
                config = AGENT_CONFIGS[agent_name]
                url = f"http://{config['host']}:{config['port']}/.well-known/agent.json"
                tasks.append(wait_for_service(agent_name, url))
            
            results = await asyncio.gather(*tasks)
            if not all(results):
                raise RuntimeError("One or more agents failed to start. Shutting down.")

        asyncio.run(wait_for_all_services())
        
        print("\n🎉 All agents are running. Press Ctrl+C to stop.")
        
        # Keep the main script alive while agents are running
        while True:
            time.sleep(1)

    except (KeyboardInterrupt, RuntimeError) as e:
        if isinstance(e, RuntimeError):
            print(f"\nError: {e}")
        print("\n🛑 Shutting down all agents...")
        
    finally:
        for agent_name, process in processes.items():
            if process.poll() is None:  # Check if process is still running
                print(f"   Terminating {agent_name} agent (PID: {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   {agent_name} agent did not terminate gracefully, killing.")
                    process.kill()
        print("✅ All agent processes terminated.")


if __name__ == "__main__":
    main()
