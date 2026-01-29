"""Worker MCP Server for inter-agent communication.

This module provides MCP tools that worker agents can use to communicate
with other agents in the same session. Workers can send messages to other
agents, receive messages, and list available agents.

These tools are designed to be used by individual agent instances
running in tmux panes, not by the orchestrator.

Usage:
    # Workers load this MCP server for inter-agent communication
    # Environment variables:
    # - AGENTIC_SESSION_ID: The current session ID
    # - AGENTIC_AGENT_ID: This worker's agent ID
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agentic.redis_client import get_client


# Debug logging
DEBUG_LOG = Path.home() / ".config" / "agentic" / "worker_mcp_debug.log"
DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

def _log_debug(msg: str) -> None:
    """Write debug message to log file."""
    with open(DEBUG_LOG, "a") as f:
        f.write(f"{time.time()}: {msg}\n")


def get_storage():
    """Get storage client."""
    return get_client(
        host=os.environ.get("AGENTIC_REDIS_HOST", "localhost"),
        port=int(os.environ.get("AGENTIC_REDIS_PORT", "6379")),
        db=int(os.environ.get("AGENTIC_REDIS_DB", "0")),
    )


def get_session_info() -> tuple[str | None, str | None]:
    """Get current session and agent ID from environment."""
    session_id = os.environ.get("AGENTIC_SESSION_ID")
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    return session_id, agent_id


# Create the worker MCP server
_log_debug("Worker MCP server initializing...")
worker_mcp = FastMCP("Agentic Worker")
_log_debug("Worker MCP server created")


@worker_mcp.tool()
def send_to_agent(
    agent_id: str = Field(description="ID of the agent to send message to (e.g., 'W1', 'W2')"),
    message: str = Field(description="The message content to send"),
) -> dict[str, Any]:
    """
    Send a message to another agent in the session.
    
    Messages are queued and the target agent will receive them when they
    call receive_message. Use this for inter-agent communication.
    
    Example:
        send_to_agent(agent_id="W1", message="My guess is 50")
    """
    session_id, my_agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not my_agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    
    # Verify target agent exists (orchestrator is a special case)
    if agent_id != "orchestrator":
        target_agent = storage.get_agent(session_id, agent_id)
        if not target_agent:
            return {"error": f"Agent {agent_id} not found in session"}
    
    # Send the message
    msg_id = storage.send_agent_message(
        session_id=session_id,
        from_agent=my_agent_id,
        to_agent=agent_id,
        message=message,
    )
    
    return {
        "status": "sent",
        "message_id": msg_id,
        "from": my_agent_id,
        "to": agent_id,
        "message_preview": message[:100] + "..." if len(message) > 100 else message,
    }


@worker_mcp.tool()
def receive_message(
    timeout: int = Field(
        default=60,
        description="Seconds to wait for a message (0 for non-blocking)",
    ),
) -> dict[str, Any]:
    """
    Receive the next message from the queue.
    
    Messages are received in order (FIFO). If no message is available,
    waits up to `timeout` seconds. Use timeout=0 for non-blocking check.
    
    Also checks if the session has been terminated via push_done_to_all.
    If session is done, returns a termination signal.
    
    Example:
        receive_message(timeout=60)  # Wait up to 60 seconds
        receive_message(timeout=0)   # Check and return immediately
    """
    session_id, agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    
    # Check if session has been marked done (via push_done_to_all or stop_session)
    if storage.is_session_done(session_id):
        return {
            "status": "session_terminated",
            "agent_id": agent_id,
            "message": "TERMINATE",
            "reason": "Session has been marked as done by orchestrator",
        }
    
    # Try to receive a message
    msg = storage.receive_agent_message(session_id, agent_id, timeout=timeout)
    
    if not msg:
        # Check session done status again after waiting
        if storage.is_session_done(session_id):
            return {
                "status": "session_terminated",
                "agent_id": agent_id,
                "message": "TERMINATE",
                "reason": "Session has been marked as done by orchestrator",
            }
        return {
            "status": "no_message",
            "agent_id": agent_id,
            "waited_seconds": timeout,
        }
    
    return {
        "status": "received",
        "message_id": msg.get("id"),
        "from": msg.get("from"),
        "message": msg.get("message"),
        "timestamp": msg.get("timestamp"),
    }


@worker_mcp.tool()
def check_messages() -> dict[str, Any]:
    """
    Check how many messages are waiting without receiving them.
    
    Use this to see if there are pending messages before blocking on receive.
    """
    session_id, agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    count = storage.get_message_count(session_id, agent_id)
    
    return {
        "agent_id": agent_id,
        "pending_messages": count,
        "has_messages": count > 0,
    }


@worker_mcp.tool()
def peek_messages(
    count: int = Field(default=5, description="Number of messages to peek at"),
) -> dict[str, Any]:
    """
    Preview pending messages without removing them from the queue.
    
    Useful to see what messages are waiting before deciding to receive them.
    """
    session_id, agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    messages = storage.peek_agent_messages(session_id, agent_id, count=count)
    
    return {
        "agent_id": agent_id,
        "messages": [
            {
                "from": m.get("from"),
                "message_preview": m.get("message", "")[:100],
                "timestamp": m.get("timestamp"),
            }
            for m in messages
        ],
        "count": len(messages),
    }


@worker_mcp.tool()
def list_agents() -> dict[str, Any]:
    """
    List all agents in the current session.
    
    Use this to discover which agents are available to communicate with.
    """
    _log_debug("list_agents: START")
    session_id, agent_id = get_session_info()
    
    if not session_id:
        _log_debug("list_agents: NO SESSION")
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    
    _log_debug(f"list_agents: getting storage for session={session_id}")
    storage = get_storage()
    _log_debug("list_agents: got storage, querying agents")
    agents = storage.get_all_agents(session_id)
    _log_debug(f"list_agents: DONE, found {len(agents)} agents")
    
    return {
        "session_id": session_id,
        "my_agent_id": agent_id,
        "agents": [
            {
                "id": a.id,
                "role": a.role,
                "status": a.status.value if hasattr(a.status, 'value') else str(a.status),
            }
            for a in agents
        ],
        "count": len(agents),
    }


@worker_mcp.tool()
def get_my_info() -> dict[str, Any]:
    """
    Get information about this agent.
    
    Returns details about the current agent including ID, role, and status.
    """
    session_id, agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    agent = storage.get_agent(session_id, agent_id)
    
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    return {
        "agent_id": agent.id,
        "role": agent.role,
        "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
        "scope": {
            "patterns": agent.scope.patterns,
            "read_only": agent.scope.read_only,
        },
        "session_id": session_id,
    }


@worker_mcp.tool()
def broadcast_message(
    message: str = Field(description="Message to send to all agents"),
    exclude_self: bool = Field(default=True, description="Exclude self from broadcast"),
) -> dict[str, Any]:
    """
    Send a message to all agents in the session.
    
    Broadcasts the message to every agent. Useful for announcements
    or when you need to reach all agents at once.
    """
    session_id, agent_id = get_session_info()
    
    if not session_id:
        return {"error": "No active session (AGENTIC_SESSION_ID not set)"}
    if not agent_id:
        return {"error": "Agent ID not set (AGENTIC_AGENT_ID not set)"}
    
    storage = get_storage()
    agents = storage.get_all_agents(session_id)
    
    sent_to = []
    for target in agents:
        if exclude_self and target.id == agent_id:
            continue
        storage.send_agent_message(
            session_id=session_id,
            from_agent=agent_id,
            to_agent=target.id,
            message=message,
        )
        sent_to.append(target.id)
    
    return {
        "status": "broadcast_sent",
        "from": agent_id,
        "sent_to": sent_to,
        "count": len(sent_to),
    }


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the worker MCP server."""
    worker_mcp.run()


if __name__ == "__main__":
    main()
