"""MCP Server interface for agentic-tmux orchestration.

This module provides a Model Context Protocol server that exposes
agentic-tmux functionality as MCP tools, resources, and prompts.
The server can be used by any MCP-compatible client (VS Code, Claude Desktop, etc).

Usage:
    # Via CLI
    agentic mcp
    
    # Or directly
    agentic-mcp
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agentic.dag import create_task_flow_diagram, validate_dag
from agentic.models import (
    Agent,
    AgenticSession,
    ExecutionPlan,
    FileScope,
    SessionStatus,
    Task,
    TaskDAG,
    TaskStatus,
)
from agentic.orchestrator import start_orchestrator_background, stop_orchestrator
from agentic.redis_client import get_client
from agentic.tmux_manager import TmuxManager, check_tmux_available


# Configuration
CONFIG_DIR = Path.home() / ".config" / "agentic"
PID_FILE = CONFIG_DIR / "orchestrator.pid"
SESSION_FILE = CONFIG_DIR / "current_session"


def get_redis_client():
    """Get storage client (Redis preferred, in-memory fallback)."""
    return get_client(
        host=os.environ.get("AGENTIC_REDIS_HOST", "localhost"),
        port=int(os.environ.get("AGENTIC_REDIS_PORT", "6379")),
        db=int(os.environ.get("AGENTIC_REDIS_DB", "0")),
    )


def get_current_session_id() -> str | None:
    """Get the current session ID if one exists."""
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text().strip()
    return os.environ.get("AGENTIC_SESSION_ID")


def save_current_session_id(session_id: str) -> None:
    """Save the current session ID."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(session_id)


def clear_current_session() -> None:
    """Clear the current session ID."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


# Create the MCP server
mcp = FastMCP("Agentic TMUX")


# =============================================================================
# MCP Tools
# =============================================================================


def _start_session_internal(
    working_dir: str,
    cli_command: str = "copilot -i",
    auto_init_hooks: bool = True,
    tmux_session: str | None = None,
) -> dict[str, Any]:
    """Internal function to start a session."""
    # Validate prerequisites
    if not check_tmux_available():
        return {"error": "tmux is not installed or not in PATH"}
    
    storage = get_redis_client()
    
    # Check for existing session
    existing_id = get_current_session_id()
    if existing_id:
        session = storage.get_session(existing_id)
        if session and session.status not in (SessionStatus.COMPLETED, SessionStatus.FAILED):
            # Return existing session info
            return {
                "session_id": existing_id,
                "status": "existing",
                "working_directory": session.working_directory,
            }
    
    # Detect current tmux session if not provided
    if not tmux_session:
        import subprocess as sp
        # First try listing attached clients - this works even from non-tmux processes
        try:
            result = sp.run(
                ["tmux", "list-clients", "-F", "#{client_session}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Get the first attached client's session
                tmux_session = result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        
        # Fallback to display-message (works if we're inside tmux)
        if not tmux_session:
            try:
                result = sp.run(
                    ["tmux", "display-message", "-p", "#S"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    tmux_session = result.stdout.strip()
            except Exception:
                pass
        
        if not tmux_session:
            tmux_session = "agentic"  # fallback
    
    # Create new session
    working_dir = os.path.abspath(working_dir)
    session = AgenticSession(working_directory=working_dir)
    session.config["cli_command"] = cli_command
    session.config["tmux_session"] = tmux_session
    
    storage.create_session(session)
    save_current_session_id(session.id)
    
    # Auto-install hooks if enabled
    hooks_status = None
    if auto_init_hooks:
        try:
            from agentic.hooks.install import install_hooks
            install_hooks(Path(working_dir))
            hooks_status = "installed"
        except Exception as e:
            hooks_status = f"failed: {e}"
    
    # Create tmux session with admin pane (use configured session name)
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    admin_pane_id = tmux.create_admin_pane(working_dir)
    session.admin_pane_id = admin_pane_id
    storage.update_session_status(session.id, SessionStatus.RUNNING)
    
    # Start orchestrator daemon
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG_DIR / f"orchestrator_{session.id}.log"
    pid = start_orchestrator_background(
        session_id=session.id,
        log_file=str(log_file),
    )
    
    if pid:
        PID_FILE.write_text(str(pid))
    
    result = {
        "session_id": session.id,
        "status": "started",
        "working_directory": working_dir,
        "cli_command": cli_command,
        "orchestrator_pid": pid,
    }
    if hooks_status:
        result["hooks"] = hooks_status
    
    return result


@mcp.tool()
def start_session(
    working_dir: str = Field(description="Working directory for the agents"),
    cli_command: str = Field(
        default="copilot -i",
        description="CLI command to use (e.g., 'copilot -i', 'claude', 'aider')",
    ),
    tmux_session: str = Field(
        default="",
        description="Tmux session name to use. If empty, creates new 'agentic' session. Set this to your current session name to add workers to your existing session.",
    ),
) -> dict[str, Any]:
    """
    Start a new agentic session.
    
    Creates agents in a tmux session. Specify tmux_session to add workers
    to your current session, or leave empty to create a separate 'agentic' session.
    """
    return _start_session_internal(
        working_dir, 
        cli_command, 
        tmux_session=tmux_session if tmux_session else None
    )


@mcp.tool()
def stop_session(
    kill_panes: bool = Field(default=True, description="Kill worker tmux panes"),
    clear_data: bool = Field(default=True, description="Clear session data from database"),
) -> dict[str, Any]:
    """
    Stop the current agentic session completely.
    
    By default, kills all worker panes and clears session data.
    Use this to clean up before starting a fresh session.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"status": "no_session", "message": "No active session"}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    killed_panes = []
    
    if session and kill_panes:
        # Kill all worker panes
        agents = redis.get_all_agents(session_id)
        tmux_session = session.config.get("tmux_session", "agentic")
        tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
        
        for agent in agents:
            if agent.pane_id:
                try:
                    tmux.kill_pane(agent.pane_id)
                    killed_panes.append(agent.pane_id)
                except Exception:
                    pass  # Pane may already be gone
    
    # Send done signal to all agents (for any still running)
    redis.push_done_to_all(session_id)
    
    # Stop orchestrator
    if PID_FILE.exists():
        stop_orchestrator(str(PID_FILE))
    
    if clear_data:
        # Delete all agents and session data
        # Note: This clears agents from the session but keeps session record
        for agent in redis.get_all_agents(session_id):
            redis.delete_agent(session_id, agent.id)
        redis.update_session_status(session_id, SessionStatus.COMPLETED)
    else:
        redis.update_session_status(session_id, SessionStatus.COMPLETED)
    
    clear_current_session()
    
    return {
        "session_id": session_id,
        "status": "stopped",
        "killed_panes": killed_panes,
        "data_cleared": clear_data,
    }


@mcp.tool()
def resume_session() -> dict[str, Any]:
    """
    Resume an existing session and get its current state.
    
    Returns session info and list of active agents.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"status": "no_session", "message": "No session to resume. Call plan_tasks to start."}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        clear_current_session()
        return {"status": "not_found", "message": "Session not found. Call plan_tasks to start."}
    
    agents = redis.get_all_agents(session_id)
    
    return {
        "session_id": session_id,
        "status": session.status.value,
        "working_directory": session.working_directory,
        "agents": [
            {"id": a.id, "role": a.role, "status": a.status.value}
            for a in agents
        ],
        "hint": "Call plan_tasks to create a new plan or get_status for details",
    }


@mcp.tool()
def plan_tasks(
    prompt: str = Field(description="Natural language description of the task to plan"),
    working_dir: str = Field(
        default=".",
        description="Working directory for the agents (starts session if needed)",
    ),
    suggested_agents: int = Field(
        default=3,
        description="Suggested number of agents to use",
    ),
) -> dict[str, Any]:
    """
    Get context and template for planning a multi-agent task.
    
    Automatically starts a session if one doesn't exist.
    Returns project file context and a template structure. You (the host model)
    should then call `create_plan` with your plan.
    """
    # Auto-start session if needed
    session_id = get_current_session_id()
    redis = get_redis_client()
    
    if not session_id:
        # Start a new session automatically
        start_result = _start_session_internal(working_dir)
        if "error" in start_result:
            return start_result
        session_id = start_result["session_id"]
    
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    # Get file context
    wd = session.working_directory
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.ts", "-o", "-name", "*.js"],
            capture_output=True,
            text=True,
            cwd=wd,
            timeout=5,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()][:100]
    except Exception:
        files = []
    
    files_summary = ", ".join(files[:20]) if files else "No files found"
    if len(files) > 20:
        files_summary += f" (+{len(files) - 20} more)"
    
    return {
        "session_id": session_id,
        "instructions": """
Create an execution plan by calling `create_plan` with agents and tasks.

Guidelines:
1. Identify natural boundaries in the work (by module, by concern, by file type)
2. Maximize parallelism where tasks are independent
3. Create explicit dependencies where order matters
4. Assign clear file scopes to prevent conflicts
5. Include review/validation tasks when appropriate
""",
        "prompt": prompt,
        "working_directory": wd,
        "files_in_scope": files_summary,
        "suggested_agents": suggested_agents,
        "next_step": "Call create_plan with your plan structure",
        "example": {
            "agents": [
                {"id": "W1", "role": "Main Developer", "scope_patterns": ["src/**"], "read_only": False},
                {"id": "W2", "role": "Test Writer", "scope_patterns": ["tests/**"], "read_only": False},
            ],
            "tasks": [
                {"id": "t1", "title": "Implement feature", "description": "...", "agent_id": "W1", "dependencies": [], "files": []},
                {"id": "t2", "title": "Write tests", "description": "...", "agent_id": "W2", "dependencies": ["t1"], "files": []},
            ],
        },
    }


@mcp.tool()
def create_plan(
    prompt: str = Field(description="Original task description"),
    agents: list[dict] = Field(
        description="List of agents with id, role, scope_patterns, and read_only"
    ),
    tasks: list[dict] = Field(
        description="List of tasks with id, title, description, agent_id, dependencies, and files"
    ),
) -> dict[str, Any]:
    """
    Create an execution plan from structured input.
    
    Use this after plan_tasks returns a planning template. The host model
    should fill out the plan structure and call this to create the plan.
    
    Example:
        create_plan(
            prompt="Add authentication",
            agents=[
                {"id": "W1", "role": "Auth Developer", "scope_patterns": ["src/auth/**"], "read_only": False}
            ],
            tasks=[
                {"id": "t1", "title": "Implement JWT", "description": "Add JWT auth", "agent_id": "W1", "dependencies": [], "files": ["src/auth/jwt.py"]}
            ]
        )
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session. Run start_session first."}
    
    redis = get_redis_client()
    
    # Build the execution plan
    plan_agents = []
    for a in agents:
        plan_agents.append(Agent(
            id=a.get("id", f"W{len(plan_agents)+1}"),
            role=a.get("role", "Worker"),
            scope=FileScope(
                patterns=a.get("scope_patterns", ["**/*"]),
                read_only=a.get("read_only", False),
            ),
        ))
    
    dag = TaskDAG()
    for t in tasks:
        task = Task(
            id=t.get("id", f"t{len(dag.tasks)+1}"),
            title=t.get("title", ""),
            description=t.get("description", ""),
            agent_id=t.get("agent_id"),
            dependencies=t.get("dependencies", []),
            files=t.get("files", []),
        )
        dag.add_task(task)
    
    execution_plan = ExecutionPlan(
        prompt=prompt,
        agents=plan_agents,
        dag=dag,
        estimated_communications=[],
    )
    
    # Validate DAG
    valid, errors = validate_dag(execution_plan.dag)
    if not valid:
        return {
            "error": "Plan validation failed",
            "validation_errors": errors,
            "hint": "Fix the issues and call create_plan again",
        }
    
    # Store plan for execution
    plan_id = f"plan_{session_id}_{int(time.time())}"
    redis.store_pending_plan(plan_id, execution_plan)
    
    return {
        "plan_id": plan_id,
        "prompt": prompt,
        "agents": [
            {
                "id": a.id,
                "role": a.role,
                "scope": {
                    "patterns": a.scope.patterns,
                    "read_only": a.scope.read_only,
                },
            }
            for a in plan_agents
        ],
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "agent_id": t.agent_id,
                "dependencies": t.dependencies,
            }
            for t in dag.tasks.values()
        ],
        "task_flow": create_task_flow_diagram(dag),
        "validation": {"is_valid": True, "errors": []},
        "hint": f"Call execute_plan with plan_id='{plan_id}' to execute this plan",
    }


@mcp.tool()
def execute_plan(
    plan_id: str = Field(description="Plan ID from plan_tasks output"),
) -> dict[str, Any]:
    """
    Execute an approved execution plan.
    
    Spawns agents in tmux panes and dispatches initial tasks.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    # Retrieve the plan
    execution_plan = redis.get_pending_plan(plan_id)
    if not execution_plan:
        return {"error": f"Plan {plan_id} not found. Create a new plan with plan_tasks."}
    
    # Validate DAG before execution
    valid, errors = validate_dag(execution_plan.dag)
    if not valid:
        return {
            "error": "Plan validation failed",
            "validation_errors": errors,
        }
    
    # Use the stored tmux session
    tmux_session = session.config.get("tmux_session", "agentic")
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    cli_command = session.config.get("cli_command", "copilot -i")
    
    # Spawn worker panes
    pane_mapping = tmux.spawn_multiple_workers(
        agents=execution_plan.agents,
        working_dir=session.working_directory,
        cli_command=cli_command,
        session_id=session_id,
    )
    
    # Register agents in Redis
    spawned_agents = []
    for agent in execution_plan.agents:
        agent.pane_id = pane_mapping.get(agent.id)
        redis.register_agent(session_id, agent)
        spawned_agents.append({
            "id": agent.id,
            "role": agent.role,
            "pane_id": agent.pane_id,
        })
    
    # Save DAG
    redis.save_dag(session_id, execution_plan.dag)
    
    # Dispatch initial tasks
    ready_tasks = execution_plan.dag.get_ready_tasks()
    dispatched_tasks = []
    for task in ready_tasks:
        if task.agent_id:
            redis.push_task(session_id, task.agent_id, task)
            redis.update_task_status(session_id, task.id, TaskStatus.PENDING)
            dispatched_tasks.append({
                "task_id": task.id,
                "agent_id": task.agent_id,
                "title": task.title,
            })
    
    # Clear the pending plan
    redis.delete_pending_plan(plan_id)
    
    return {
        "status": "executing",
        "spawned_agents": spawned_agents,
        "dispatched_tasks": dispatched_tasks,
        "total_tasks": len(execution_plan.dag.tasks),
        "monitor_hint": "Use get_status to monitor progress",
    }


def _build_communication_instructions(agent_id: str, other_agents: list[str]) -> str:
    """Build inter-agent communication instructions to include in agent prompts."""
    agents_list = ", ".join(other_agents) if other_agents else "none yet (use list_agents to discover)"
    return f"""

## Inter-Agent Communication

You are agent **{agent_id}**. Other agents in this session: {agents_list}

You have MCP tools for inter-agent communication:

- **send_to_agent(agent_id, message)** - Send a message to another agent (or "orchestrator")
- **receive_message(timeout=30)** - Wait for and receive the next message
- **check_messages()** - Check if there are pending messages (non-blocking)
- **list_agents()** - List all agents in this session
- **broadcast_message(message)** - Send to all agents

**CRITICAL: MESSAGE POLLING LOOP**
You MUST keep polling for messages until your task terminates:
1. Call `receive_message(timeout=60)` to wait for messages
2. If `status` is "received", process the message and respond via `send_to_agent`
3. If `status` is "no_message", call `receive_message` again immediately
4. KEEP CALLING `receive_message` - do NOT stop polling until you receive a termination message
5. Do NOT wait for user input - stay in the polling loop

To report results to the orchestrator, use: `send_to_agent(agent_id="orchestrator", message="...")`
"""


@mcp.tool()
def spawn_agent(
    role: str = Field(description="Role description for the agent (used as initial prompt)"),
    scope_patterns: list[str] = Field(
        default=["**/*"],
        description="File patterns the agent can access",
    ),
    read_only: bool = Field(
        default=False,
        description="Whether the agent has read-only access",
    ),
    initial_task: str = Field(
        default="",
        description="Optional initial task (if empty, role is used as the initial prompt)",
    ),
    enable_communication: bool = Field(
        default=True,
        description="Add inter-agent communication instructions to the prompt (default: True)",
    ),
    wait_ready: bool = Field(
        default=True,
        description="Wait for agent to initialize before returning (checks for CLI readiness)",
    ),
    wait_timeout: int = Field(
        default=60,
        description="Seconds to wait for agent to be ready",
    ),
) -> dict[str, Any]:
    """
    Spawn a single new agent in a tmux pane.
    
    The agent starts with the `role` as its initial prompt (or `initial_task` if provided).
    By default, adds inter-agent communication MCP tools (send_to_agent, receive_message, etc.).
    
    The role should contain the full system prompt/instructions for the agent.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    # Get existing agents to determine next ID
    existing_agents = redis.get_all_agents(session_id)
    next_id = f"W{len(existing_agents) + 1}"
    other_agent_ids = [a.id for a in existing_agents]
    
    # Build prompt with optional communication instructions
    agent_prompt = initial_task if initial_task else role
    if enable_communication:
        comm_instructions = _build_communication_instructions(next_id, other_agent_ids)
        agent_prompt = agent_prompt + comm_instructions
    
    # Create agent
    agent = Agent(
        id=next_id,
        role=role,
        scope=FileScope(patterns=scope_patterns, read_only=read_only),
    )
    
    # Spawn pane using stored tmux session
    tmux_session = session.config.get("tmux_session", "agentic")
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    cli_command = session.config.get("cli_command", "copilot -i")
    
    pane_id = tmux.spawn_worker_pane(
        agent=agent,
        working_dir=session.working_directory,
        cli_command=cli_command,
        session_id=session_id,
        initial_task=agent_prompt,
    )
    
    agent.pane_id = pane_id
    redis.register_agent(session_id, agent)
    
    result = {
        "agent_id": agent.id,
        "role": role[:100] + "..." if len(role) > 100 else role,  # Truncate for readability
        "pane_id": pane_id,
        "scope": {
            "patterns": agent.scope.patterns,
            "read_only": agent.scope.read_only,
        },
    }
    
    # Wait for agent to be ready if requested
    if wait_ready:
        ready_status = _wait_for_agent_ready(tmux, pane_id, timeout=wait_timeout)
        result["ready"] = ready_status["ready"]
        result["wait_time"] = ready_status.get("elapsed_seconds", 0)
        if not ready_status["ready"]:
            result["ready_warning"] = "Agent may not be fully initialized - check pane output"
    
    return result


def _wait_for_agent_ready(
    tmux: TmuxManager,
    pane_id: str,
    timeout: int = 60,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """
    Internal function to wait for an agent to be ready.
    
    Looks for signs the CLI has loaded (prompt indicator or output activity).
    """
    import hashlib
    
    start_time = time.time()
    initial_output = tmux.capture_pane_output(pane_id, lines=50)
    initial_hash = hashlib.md5(initial_output.encode()).hexdigest()
    
    # Wait for output to change (indicating CLI has started processing)
    elapsed = 0.0
    stable_count = 0
    last_hash = initial_hash
    
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        
        current_output = tmux.capture_pane_output(pane_id, lines=50)
        current_hash = hashlib.md5(current_output.encode()).hexdigest()
        
        # Check for readiness indicators in output
        output_lower = current_output.lower()
        
        # Look for common CLI readiness indicators
        ready_indicators = [
            "ready",
            "listening",
            "started",
            "initialized",
            "waiting for input",
            ">",  # Common prompt character
            "copilot",  # CLI name
        ]
        
        # If output changed and has any indicator, consider ready
        if current_hash != initial_hash:
            for indicator in ready_indicators:
                if indicator in output_lower:
                    return {
                        "ready": True,
                        "elapsed_seconds": round(elapsed, 1),
                        "indicator": indicator,
                    }
            
            # If output has stabilized (same for 2 polls), consider ready
            if current_hash == last_hash:
                stable_count += 1
                if stable_count >= 2:
                    return {
                        "ready": True,
                        "elapsed_seconds": round(elapsed, 1),
                        "reason": "output_stabilized",
                    }
            else:
                stable_count = 0
        
        last_hash = current_hash
    
    return {
        "ready": False,
        "elapsed_seconds": round(elapsed, 1),
        "reason": "timeout",
    }


@mcp.tool()
def send_task(
    agent_id: str = Field(description="ID of the agent to send the task to"),
    task_description: str = Field(description="Description of the task"),
    files: list[str] = Field(
        default=[],
        description="List of files relevant to the task",
    ),
    force: bool = Field(
        default=False,
        description="Force sending even if a task is already running or queued",
    ),
) -> dict[str, Any]:
    """
    Send a task to a specific agent.
    
    By default, will not send if the agent already has a running task or queued tasks.
    Use force=True to bypass this check.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    
    # Verify agent exists
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    # Check if agent already has tasks (unless force=True)
    if not force:
        if agent.current_task_id:
            return {
                "error": f"Agent {agent_id} is already running a task",
                "current_task_id": agent.current_task_id,
                "hint": "Use force=True to queue anyway, or wait for the task to complete",
            }
        if agent.task_queue_length > 0:
            return {
                "error": f"Agent {agent_id} already has {agent.task_queue_length} queued task(s)",
                "hint": "Use force=True to queue anyway, or wait for tasks to complete",
            }
    
    task = Task(
        title=task_description[:50],
        description=task_description,
        from_agent="mcp_client",
        files=files,
    )
    
    redis.push_task(session_id, agent_id, task)
    
    return {
        "task_id": task.id,
        "agent_id": agent_id,
        "title": task.title,
        "status": "queued",
    }


@mcp.tool()
def get_status() -> dict[str, Any]:
    """
    Get the current status of all agents and tasks.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    agents = redis.get_all_agents(session_id)
    current_time = time.time()
    
    agent_status = []
    for agent in agents:
        heartbeat_age = int(current_time - agent.last_heartbeat)
        agent_status.append({
            "id": agent.id,
            "role": agent.role,
            "status": agent.status.value,
            "current_task": agent.current_task_id,
            "queue_length": agent.task_queue_length,
            "heartbeat_age_seconds": heartbeat_age,
            "healthy": heartbeat_age < 120,
        })
    
    # Get DAG progress
    dag = redis.get_dag(session_id)
    if dag:
        completed, total = dag.get_completion_progress()
        progress = {
            "completed_tasks": completed,
            "total_tasks": total,
            "percentage": round(completed / total * 100, 1) if total > 0 else 0,
        }
    else:
        progress = {"completed_tasks": 0, "total_tasks": 0, "percentage": 0}
    
    return {
        "session_id": session_id,
        "session_status": session.status.value,
        "working_directory": session.working_directory,
        "agents": agent_status,
        "progress": progress,
    }


@mcp.tool()
def get_agent_logs(
    agent_id: str = Field(description="ID of the agent"),
    count: int = Field(default=50, description="Number of log entries to retrieve"),
) -> dict[str, Any]:
    """
    Get recent logs for a specific agent.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    logs = redis.get_agent_logs(session_id, agent_id, count=count)
    
    return {
        "agent_id": agent_id,
        "logs": [
            {
                "timestamp": time.strftime("%H:%M:%S", time.localtime(log.timestamp)),
                "action": log.action,
                "file": log.file,
                "tool": log.tool,
            }
            for log in logs
        ],
    }


@mcp.tool()
def clear_agents() -> dict[str, Any]:
    """
    Clear all worker agents but keep the session.
    
    Useful for starting fresh with a new plan.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    
    # Use stored tmux session
    tmux_session = session.config.get("tmux_session", "agentic") if session else "agentic"
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    
    # Send done signal
    redis.push_done_to_all(session_id)
    
    # Kill worker panes
    killed = tmux.kill_all_workers()
    
    # Clear agents from Redis
    agents = redis.get_all_agents(session_id)
    for agent in agents:
        redis.delete_agent(session_id, agent.id)
    
    # Clear DAG
    redis.save_dag(session_id, TaskDAG())
    
    return {
        "killed_panes": killed,
        "cleared_agents": len(agents),
        "status": "cleared",
    }


# =============================================================================
# Inter-Agent Communication Tools
# =============================================================================


@mcp.tool()
def send_message(
    agent_id: str = Field(description="ID of the agent to send the message to"),
    message: str = Field(description="Message to send to the agent"),
    from_agent: str = Field(default="orchestrator", description="Sender ID (defaults to 'orchestrator')"),
) -> dict[str, Any]:
    """
    Send a message to an agent's message queue.
    
    The message is queued and the agent will receive it when they call
    receive_message(). This is the primary way for the orchestrator to
    communicate with agents.
    
    Since Copilot CLI doesn't accept tmux send-keys, this uses the same
    message queue that agents poll via MCP tools.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    # Send message via the queue (same as worker MCP send_to_agent)
    msg_id = redis.send_agent_message(
        session_id=session_id,
        from_agent=from_agent,
        to_agent=agent_id,
        message=message,
    )
    
    return {
        "status": "queued",
        "message_id": msg_id,
        "from": from_agent,
        "to": agent_id,
        "message_preview": message[:100] + "..." if len(message) > 100 else message,
    }


@mcp.tool()
def receive_message_from_agents(
    timeout: int = Field(default=60, description="Seconds to wait for a message"),
) -> dict[str, Any]:
    """
    Receive a message sent to the orchestrator from any agent.
    
    Agents can send messages to 'orchestrator' using send_to_agent.
    Use this to receive those messages and monitor game progress.
    
    Returns the first message in the queue, or indicates no message after timeout.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    
    # Receive message addressed to 'orchestrator'
    msg = redis.receive_agent_message(session_id, "orchestrator", timeout=timeout)
    
    if not msg:
        return {
            "status": "no_message",
            "waited_seconds": timeout,
        }
    
    return {
        "status": "received",
        "message_id": msg.get("id"),
        "from": msg.get("from"),
        "message": msg.get("message"),
        "timestamp": msg.get("timestamp"),
    }


@mcp.tool()
def check_orchestrator_messages() -> dict[str, Any]:
    """
    Check how many messages are waiting for the orchestrator (non-blocking).
    
    Use this to see if any agents have sent messages without blocking.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    count = redis.get_message_count(session_id, "orchestrator")
    
    # Also peek at the messages
    messages = redis.peek_agent_messages(session_id, "orchestrator", count=5)
    
    return {
        "pending_messages": count,
        "has_messages": count > 0,
        "preview": [
            {
                "from": m.get("from"),
                "message_preview": m.get("message", "")[:100],
            }
            for m in messages
        ],
    }


@mcp.tool()
def read_pane_output(
    agent_id: str = Field(description="ID of the agent to read output from"),
    lines: int = Field(default=50, description="Number of lines to capture"),
) -> dict[str, Any]:
    """
    Read recent output from an agent's tmux pane.
    
    Use this to see what an agent has outputted, including responses
    to messages you've sent.
    """
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    if not agent.pane_id:
        return {"error": f"Agent {agent_id} has no pane"}
    
    session = redis.get_session(session_id)
    tmux_session = session.config.get("tmux_session", "agentic") if session else "agentic"
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    
    output = tmux.capture_pane_output(agent.pane_id, lines=lines)
    
    return {
        "agent_id": agent_id,
        "output": output,
        "lines_captured": len(output.split('\n')),
    }


@mcp.tool()
def wait_for_output(
    agent_id: str = Field(description="ID of the agent to wait for"),
    timeout: int = Field(default=30, description="Maximum seconds to wait"),
    poll_interval: float = Field(default=2.0, description="Seconds between checks"),
) -> dict[str, Any]:
    """
    Wait for an agent to produce new output.
    
    Polls the agent's pane output until it changes or timeout is reached.
    Use after sending a message to wait for the agent's response.
    """
    import hashlib
    
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    if not agent.pane_id:
        return {"error": f"Agent {agent_id} has no pane"}
    
    session = redis.get_session(session_id)
    tmux_session = session.config.get("tmux_session", "agentic") if session else "agentic"
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    
    # Get initial output hash
    initial_output = tmux.capture_pane_output(agent.pane_id, lines=100)
    initial_hash = hashlib.md5(initial_output.encode()).hexdigest()
    
    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        current_output = tmux.capture_pane_output(agent.pane_id, lines=100)
        current_hash = hashlib.md5(current_output.encode()).hexdigest()
        
        if current_hash != initial_hash:
            return {
                "agent_id": agent_id,
                "status": "output_changed",
                "elapsed_seconds": elapsed,
                "output": current_output,
            }
    
    return {
        "agent_id": agent_id,
        "status": "timeout",
        "elapsed_seconds": elapsed,
        "output": tmux.capture_pane_output(agent.pane_id, lines=100),
    }


@mcp.tool()
def wait_for_agent_ready(
    agent_id: str = Field(description="ID of the agent to wait for"),
    timeout: int = Field(default=120, description="Maximum seconds to wait"),
    completion_phrases: list[str] = Field(
        default=[],
        description="Phrases to look for indicating the agent has finished responding (e.g., 'higher', 'lower', 'correct')",
    ),
) -> dict[str, Any]:
    """
    Wait for an agent to finish its current task/response.
    
    Use this after the agent has received a prompt to wait until it has finished
    generating a response. Looks for output stabilization or specific completion phrases.
    
    For the guess-the-number game, use completion_phrases=['higher', 'lower', 'correct']
    to detect when the Chooser has responded.
    """
    import hashlib
    
    session_id = get_current_session_id()
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client()
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    if not agent.pane_id:
        return {"error": f"Agent {agent_id} has no pane"}
    
    session = redis.get_session(session_id)
    tmux_session = session.config.get("tmux_session", "agentic") if session else "agentic"
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    
    start_time = time.time()
    poll_interval = 2.0
    stable_count = 0
    last_hash = ""
    
    while (time.time() - start_time) < timeout:
        current_output = tmux.capture_pane_output(agent.pane_id, lines=100)
        current_hash = hashlib.md5(current_output.encode()).hexdigest()
        
        # Check for completion phrases in output
        output_lower = current_output.lower()
        for phrase in completion_phrases:
            if phrase.lower() in output_lower:
                elapsed = time.time() - start_time
                return {
                    "agent_id": agent_id,
                    "status": "completion_phrase_found",
                    "phrase": phrase,
                    "elapsed_seconds": round(elapsed, 1),
                    "output": current_output,
                }
        
        # Check for output stabilization (same hash for 3 consecutive polls)
        if current_hash == last_hash:
            stable_count += 1
            if stable_count >= 3:
                elapsed = time.time() - start_time
                return {
                    "agent_id": agent_id,
                    "status": "output_stabilized",
                    "elapsed_seconds": round(elapsed, 1),
                    "output": current_output,
                }
        else:
            stable_count = 0
        
        last_hash = current_hash
        time.sleep(poll_interval)
    
    elapsed = time.time() - start_time
    return {
        "agent_id": agent_id,
        "status": "timeout",
        "elapsed_seconds": round(elapsed, 1),
        "output": tmux.capture_pane_output(agent.pane_id, lines=100),
    }


@mcp.tool()
def init_hooks(
    repo_path: str = Field(
        default=".",
        description="Path to the repository to initialize hooks in",
    ),
) -> dict[str, Any]:
    """
    Initialize agentic hooks in a repository.
    
    Creates .github/hooks/ with session lifecycle hooks that enable
    agent registration, file scope validation, and action logging.
    """
    from agentic.hooks.install import install_hooks
    
    try:
        path = Path(repo_path).resolve()
        install_hooks(path)
        hooks_dir = path / ".github" / "hooks"
        return {
            "status": "installed",
            "hooks_directory": str(hooks_dir),
            "hooks": [
                "sessionStart.json - Agent registration",
                "preToolUse.json - File scope validation",
                "postToolUse.json - Action logging",
                "sessionEnd.json - Cleanup",
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# MCP Resources
# =============================================================================


@mcp.resource("session://status")
def get_session_status_resource() -> str:
    """Current session status as JSON."""
    session_id = get_current_session_id()
    if not session_id:
        return json.dumps({"error": "No active session"})
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        return json.dumps({"error": "Session not found"})
    
    return json.dumps({
        "session_id": session_id,
        "status": session.status.value,
        "working_directory": session.working_directory,
        "created_at": session.created_at,
    }, indent=2)


@mcp.resource("dag://current")
def get_dag_resource() -> str:
    """Current task DAG visualization."""
    session_id = get_current_session_id()
    if not session_id:
        return "No active session"
    
    redis = get_redis_client()
    dag = redis.get_dag(session_id)
    if not dag:
        return "No DAG defined"
    
    return create_task_flow_diagram(dag)


@mcp.resource("agents://list")
def get_agents_resource() -> str:
    """List of all agents in the current session."""
    session_id = get_current_session_id()
    if not session_id:
        return json.dumps({"error": "No active session"})
    
    redis = get_redis_client()
    agents = redis.get_all_agents(session_id)
    
    return json.dumps([
        {
            "id": a.id,
            "role": a.role,
            "status": a.status.value,
            "scope": {
                "patterns": a.scope.patterns,
                "read_only": a.scope.read_only,
            },
        }
        for a in agents
    ], indent=2)


# =============================================================================
# MCP Prompts
# =============================================================================


@mcp.prompt()
def orchestrate_task(
    task: str = Field(description="The task to orchestrate"),
    working_dir: str = Field(default=".", description="Working directory"),
) -> str:
    """
    Generate a prompt for orchestrating a multi-agent task.
    """
    return f"""Orchestrate this task using multiple AI coding agents:

Task: {task}
Working Directory: {working_dir}

Steps:

1. **Plan** - Call `plan_tasks(prompt="{task}", working_dir="{working_dir}")`
   This auto-starts a session and returns project context.

2. **Create Plan** - Call `create_plan` with:
   - agents: List with id, role, scope_patterns (and optionally read_only)
   - tasks: List with id, title, description, agent_id, dependencies

3. **Execute** - Call `execute_plan(plan_id="...")`

4. **Monitor** - Call `get_status()` to track progress

5. **Complete** - Call `stop_session()` when done

Tips:
- Maximize parallelism - independent tasks can run concurrently
- Use clear file scopes to prevent conflicts between agents
- Include a reviewer agent for validation tasks
"""


@mcp.prompt()
def review_agent_work(
    agent_id: str = Field(description="The agent ID to review"),
) -> str:
    """
    Generate a prompt for reviewing an agent's work.
    """
    return f"""Review the work done by agent {agent_id}.

1. First, get the agent's logs:
   Call `get_agent_logs` with agent_id="{agent_id}"

2. Check the current status:
   Call `get_status` to see if the agent has pending tasks

3. Based on the logs, evaluate:
   - What tasks did the agent complete?
   - Were there any errors or issues?
   - Did the agent stay within its file scope?
   - Should any work be redone or extended?

4. If needed, send follow-up tasks:
   Use `send_task` to assign corrections or additional work
"""


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
