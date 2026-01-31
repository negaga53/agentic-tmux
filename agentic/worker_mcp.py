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

from agentic.config import (
    get_debug_log,
    ensure_config_dir,
    WORKING_DIR_ENV_VAR,
)
from agentic.redis_client import get_client
from agentic.monitor import log_activity


def _get_working_dir() -> str:
    """Get working directory from environment or CWD."""
    return os.environ.get(WORKING_DIR_ENV_VAR) or os.getcwd()


def _log_debug(msg: str) -> None:
    """Write debug message to log file."""
    working_dir = _get_working_dir()
    debug_log = get_debug_log(working_dir, "worker_mcp_debug")
    ensure_config_dir(working_dir)
    with open(debug_log, "a") as f:
        f.write(f"{time.time()}: {msg}\n")


def get_storage():
    """Get storage client."""
    working_dir = _get_working_dir()
    return get_client(
        host=os.environ.get("AGENTIC_REDIS_HOST", "localhost"),
        port=int(os.environ.get("AGENTIC_REDIS_PORT", "6379")),
        db=int(os.environ.get("AGENTIC_REDIS_DB", "0")),
        working_dir=working_dir,
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
    agent_id: str = Field(
        description="Target agent ID (e.g., 'W1', 'W2', 'orchestrator'). Use 'orchestrator' to report results to the coordinator."
    ),
    message: str = Field(
        description="Complete message content. Include all relevant data - results, status, errors. This is the ONLY way to deliver information."
    ),
) -> dict[str, Any]:
    """
    Send a message to another agent or the orchestrator. This is your ONLY method
    to deliver results - text output in your response is NOT seen by other agents.
    
    CRITICAL: You MUST call this with agent_id="orchestrator" to report your final
    results. If you don't call this, the orchestrator will never receive your work
    and the session will hang or fail. Always include complete results in the message.
    
    Use this after completing your task to report results, or to coordinate with
    other workers during execution. Messages are queued and delivered when the
    recipient calls receive_message().
    
    Example - reporting results:
        send_to_agent(agent_id="orchestrator", message='{"status": "complete", "result": "analysis done"}')
    
    Example - coordinating with another worker:
        send_to_agent(agent_id="W2", message="Ready for you to start tests")
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
    
    # Log message sent
    log_activity("message_sent", {
        "from": my_agent_id,
        "to": agent_id,
        "message_preview": message[:500],
    }, session_id=session_id, working_dir=_get_working_dir())
    
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
        default=300,
        description="Seconds to wait for a message. Use 300 (5 min) for normal polling, 0 for non-blocking check.",
    ),
) -> dict[str, Any]:
    """
    Wait for and receive the next incoming message. You MUST call this in a loop
    after reporting your results to stay available for follow-up instructions.
    
    Returns one of three statuses:
    - "received": A message arrived. Process it and respond if needed.
    - "no_message": No message within timeout. DO NOT EXIT - call again to keep polling.
    - "session_terminated": Session is ending. Exit gracefully.
    
    CRITICAL: If status is "no_message", you MUST call receive_message again.
    Do NOT exit or end your turn. The orchestrator may send follow-up work at any time.
    Only exit when status is "session_terminated" or message contains "TERMINATE".
    
    Example polling loop:
        while True:
            msg = receive_message(timeout=300)
            if msg["status"] == "session_terminated": break
            if msg["status"] == "received":
                if "TERMINATE" in msg["message"]: break
                # process message and respond
            # "no_message" -> continue loop, do NOT exit
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
    
    # Log message received
    log_activity("message_received", {
        "agent_id": agent_id,
        "from": msg.get("from"),
        "message_preview": msg.get("message", "")[:50],
    }, session_id=session_id, working_dir=_get_working_dir())
    
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
    Check how many messages are waiting without blocking or removing them.
    
    Use this for a quick, non-blocking check before starting a long operation.
    Returns the count of pending messages. This does NOT remove messages from
    the queue - use receive_message() to actually get and process them.
    
    This is optional - you can also just call receive_message(timeout=0) for
    a similar non-blocking check that also retrieves the first message.
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
    Discover all agents in this multi-agent session. CALL THIS FIRST before
    doing any other work. Returns the IDs and roles of all workers.
    
    This tells you who else is working in the session, so you can coordinate.
    Without calling this, you won't know which agent IDs are valid for
    send_to_agent(). The orchestrator also needs you to call this to confirm
    you've properly initialized.
    
    Returns your own agent ID in 'my_agent_id' and a list of all agents
    including yourself. Use this to understand the team structure before
    starting your task.
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
    message: str = Field(description="Message to send to all agents. Include complete information as this goes to everyone."),
    exclude_self: bool = Field(default=True, description="Whether to exclude yourself from the broadcast (default: True)"),
) -> dict[str, Any]:
    """
    Send a message to ALL agents in the session at once. Use this for
    session-wide announcements that everyone needs to know about.
    
    This is a convenience method that calls send_to_agent() for each agent.
    For targeted communication to specific agents, use send_to_agent() directly.
    By default, you won't receive your own broadcast (exclude_self=True).
    
    Example: Announcing completion of a shared dependency:
        broadcast_message(message="Shared library update complete, you can proceed")
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
