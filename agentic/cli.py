"""CLI interface for agentic-tmux.

The CLI provides debugging and monitoring commands. For planning and execution,
use the MCP server via `agentic mcp` which integrates with VS Code, Claude Desktop,
and other MCP clients.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from agentic.models import (
    SessionStatus,
    Task,
)
from agentic.orchestrator import stop_orchestrator
from agentic.redis_client import get_client
from agentic.tmux_manager import TmuxManager

console = Console()

# Config paths
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


def clear_current_session() -> None:
    """Clear the current session ID."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Agentic TMUX - Multi-agent orchestration for CLI coding assistants.
    
    Primary interface is the MCP server. Start it with: agentic mcp
    
    These CLI commands are for debugging and monitoring.
    """
    pass


@main.command()
@click.option("--transport", "-t", default="stdio", help="Transport type (stdio, sse, streamable-http)")
def mcp(transport: str):
    """Start the MCP server for integration with MCP clients.
    
    This is the primary interface for orchestrating agents.
    """
    from agentic.mcp_server import mcp as mcp_server
    
    console.print(f"[cyan]Starting MCP server with {transport} transport...[/cyan]")
    mcp_server.run(transport=transport)


@main.command()
@click.option("--watch", "-w", is_flag=True, help="Watch mode (updates every 2s)")
def status(watch: bool):
    """Show status of all agents and tasks."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_redis_client()
    
    if watch:
        with Live(console=console, refresh_per_second=0.5) as live:
            while True:
                try:
                    live.update(render_status(session_id, redis))
                    time.sleep(2)
                except KeyboardInterrupt:
                    break
    else:
        console.print(render_status(session_id, redis))


def render_status(session_id: str, storage: Any) -> Panel:
    """Render status panel."""
    session = storage.get_session(session_id)
    if not session:
        return Panel("[red]Session not found[/red]")
    
    # Agents table
    agent_table = Table(title="")
    agent_table.add_column("Pane", style="dim")
    agent_table.add_column("Role")
    agent_table.add_column("Status")
    agent_table.add_column("Current Task")
    agent_table.add_column("Queue")
    agent_table.add_column("❤️", justify="right")
    
    agents = storage.get_all_agents(session_id)
    for agent in agents:
        status_str = agent.status.value
        if agent.status.value == "working":
            status_str = f"[green]{status_str}[/green]"
        elif agent.status.value == "idle":
            status_str = f"[dim]{status_str}[/dim]"
        elif agent.status.value.startswith("waiting"):
            status_str = f"[yellow]{status_str}[/yellow]"
        
        heartbeat_age = int(time.time() - agent.last_heartbeat)
        heartbeat_str = f"{heartbeat_age}s" if heartbeat_age < 120 else f"[red]{heartbeat_age}s[/red]"
        
        agent_table.add_row(
            agent.id,
            agent.role,
            status_str,
            agent.current_task_id or "-",
            str(agent.task_queue_length),
            heartbeat_str,
        )
    
    # Progress
    dag = storage.get_dag(session_id)
    if dag:
        completed, total = dag.get_completion_progress()
        progress_bar = "█" * completed + "░" * (total - completed)
        progress_str = f"{progress_bar} {completed}/{total} tasks complete"
    else:
        progress_str = "No tasks"
    
    content = f"{agent_table}\n\n[bold]Progress:[/bold] {progress_str}"
    
    return Panel(
        content,
        title=f"AGENTIC SESSION: {session_id}",
        subtitle=f"Status: {session.status.value}",
        border_style="blue",
    )


@main.command()
@click.argument("agent_id")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def logs(agent_id: str, follow: bool, lines: int):
    """View logs for a specific agent."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_redis_client()
    
    # Get initial logs
    log_entries = redis.get_agent_logs(session_id, agent_id, count=lines)
    
    for log in log_entries:
        ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
        file_str = f" ({log.file})" if log.file else ""
        console.print(f"[dim]{ts}[/dim] [{log.agent_id}] {log.action}{file_str}")
    
    if follow:
        last_id = "0" if not log_entries else str(int(log_entries[-1].timestamp * 1000))
        console.print("[dim]Waiting for new logs... (Ctrl+C to stop)[/dim]")
        
        while True:
            try:
                new_logs = redis.get_agent_logs(session_id, agent_id, count=10, last_id=last_id)
                for log in new_logs:
                    ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
                    file_str = f" ({log.file})" if log.file else ""
                    console.print(f"[dim]{ts}[/dim] [{log.agent_id}] {log.action}{file_str}")
                if new_logs:
                    last_id = str(int(new_logs[-1].timestamp * 1000))
                time.sleep(1)
            except KeyboardInterrupt:
                break


@main.command()
@click.argument("agent_id")
@click.argument("task_description")
def send(agent_id: str, task_description: str):
    """Send a task to a specific agent."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[red]Error:[/red] No active session")
        sys.exit(1)
    
    redis = get_redis_client()
    
    task = Task(
        title=task_description[:50],
        description=task_description,
        from_agent="cli",
    )
    
    redis.push_task(session_id, agent_id, task)
    console.print(f"[green]✓[/green] Task sent to {agent_id}")


@main.command()
def stop():
    """Stop the current agentic session."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_redis_client()
    
    # Send done signal to all agents
    redis.push_done_to_all(session_id)
    
    # Stop orchestrator
    if PID_FILE.exists():
        stop_orchestrator(str(PID_FILE))
        console.print("[green]✓[/green] Stopped orchestrator")
    
    # Kill tmux session
    tmux = TmuxManager()
    if tmux.session_exists():
        if Confirm.ask("Kill tmux session?"):
            tmux.kill_session()
            console.print("[green]✓[/green] Killed tmux session")
    
    # Update session status
    redis.update_session_status(session_id, SessionStatus.COMPLETED)
    clear_current_session()
    
    console.print(f"[green]✓[/green] Session [cyan]{session_id}[/cyan] stopped")


@main.command()
@click.option("--force", "-f", is_flag=True, help="Force clear without confirmation")
def clear(force: bool):
    """Clear all workers but keep the session."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    if not force and not Confirm.ask("Kill all worker panes?"):
        return
    
    redis = get_redis_client()
    tmux = TmuxManager()
    
    # Send done signal
    redis.push_done_to_all(session_id)
    
    # Kill worker panes
    killed = tmux.kill_all_workers()
    console.print(f"[green]✓[/green] Killed {killed} worker panes")
    
    # Clear agents from Redis
    agents = redis.get_all_agents(session_id)
    for agent in agents:
        redis.delete_agent(session_id, agent.id)
    
    console.print("[green]✓[/green] Session cleared")


@main.command()
@click.option("--output", "-o", default="session_export.json", help="Output file")
def export(output: str):
    """Export session transcript and state."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    import json
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    
    if not session:
        console.print("[red]Error:[/red] Session not found")
        return
    
    # Collect all data
    export_data = {
        "session": session.to_config(),
        "agents": [],
        "dag": None,
        "logs": {},
    }
    
    agents = redis.get_all_agents(session_id)
    for agent in agents:
        export_data["agents"].append(agent.to_config())
        log_entries = redis.get_agent_logs(session_id, agent.id, count=1000)
        export_data["logs"][agent.id] = [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "file": log.file,
                "tool": log.tool,
            }
            for log in log_entries
        ]
    
    dag = redis.get_dag(session_id)
    if dag:
        export_data["dag"] = dag.to_dict()
    
    # Write to file
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)
    
    console.print(f"[green]✓[/green] Exported to {output}")


# =============================================================================
# Inter-agent messaging commands (for use by workers)
# =============================================================================


@main.command(name="msg-send")
@click.argument("to_agent")
@click.argument("message")
def msg_send(to_agent: str, message: str):
    """Send a message to another agent.
    
    Workers can use this to communicate with other agents.
    Requires AGENTIC_SESSION_ID and AGENTIC_AGENT_ID environment variables.
    """
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id()
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session (set AGENTIC_SESSION_ID)")
        sys.exit(1)
    if not agent_id:
        console.print("[red]Error:[/red] Agent ID not set (set AGENTIC_AGENT_ID)")
        sys.exit(1)
    
    storage = get_redis_client()
    
    # Verify target exists
    target = storage.get_agent(session_id, to_agent)
    if not target:
        console.print(f"[red]Error:[/red] Agent {to_agent} not found")
        sys.exit(1)
    
    # Send message
    msg_id = storage.send_agent_message(session_id, agent_id, to_agent, message)
    console.print(f"[green]✓[/green] Message sent to {to_agent} (id: {msg_id})")


@main.command(name="msg-recv")
@click.option("--timeout", "-t", default=30, help="Seconds to wait for message")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def msg_recv(timeout: int, raw: bool):
    """Receive the next message from the queue.
    
    Workers use this to receive messages from other agents.
    Blocks until a message arrives or timeout is reached.
    """
    import json
    
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id()
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session (set AGENTIC_SESSION_ID)")
        sys.exit(1)
    if not agent_id:
        console.print("[red]Error:[/red] Agent ID not set (set AGENTIC_AGENT_ID)")
        sys.exit(1)
    
    storage = get_redis_client()
    msg = storage.receive_agent_message(session_id, agent_id, timeout=timeout)
    
    if not msg:
        if raw:
            console.print("{}")
        else:
            console.print("[yellow]No message received[/yellow]")
        sys.exit(0)
    
    if raw:
        console.print(json.dumps(msg))
    else:
        console.print(f"[cyan]From:[/cyan] {msg.get('from')}")
        console.print(f"[cyan]Message:[/cyan] {msg.get('message')}")


@main.command(name="msg-list")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def msg_list(raw: bool):
    """List available agents and pending message count.
    
    Shows all agents in the session and how many messages are waiting.
    """
    import json
    
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id()
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session")
        sys.exit(1)
    
    storage = get_redis_client()
    agents = storage.get_all_agents(session_id)
    
    if raw:
        data = {
            "my_id": agent_id,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "pending_messages": storage.get_message_count(session_id, a.id),
                }
                for a in agents
            ],
        }
        console.print(json.dumps(data))
    else:
        console.print(f"[cyan]Your ID:[/cyan] {agent_id or 'not set'}")
        console.print(f"[cyan]Session:[/cyan] {session_id}")
        console.print()
        
        table = Table(title="Agents")
        table.add_column("ID")
        table.add_column("Role")
        table.add_column("Pending Messages", justify="right")
        
        for agent in agents:
            msg_count = storage.get_message_count(session_id, agent.id)
            is_me = " (me)" if agent.id == agent_id else ""
            table.add_row(
                f"{agent.id}{is_me}",
                agent.role[:40],
                str(msg_count),
            )
        
        console.print(table)


@main.command(name="worker-mcp")
@click.option("--transport", "-t", default="stdio", help="Transport type")
def worker_mcp_cmd(transport: str):
    """Start the worker MCP server for inter-agent communication.
    
    Workers can use this MCP server to send/receive messages to other agents.
    Requires AGENTIC_SESSION_ID and AGENTIC_AGENT_ID environment variables.
    """
    from agentic.worker_mcp import worker_mcp
    
    session_id = os.environ.get("AGENTIC_SESSION_ID")
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id or not agent_id:
        console.print("[yellow]Warning:[/yellow] AGENTIC_SESSION_ID or AGENTIC_AGENT_ID not set")
        console.print("Make sure these are set for full functionality")
    
    console.print(f"[cyan]Starting Worker MCP server ({transport})...[/cyan]")
    worker_mcp.run(transport=transport)


if __name__ == "__main__":
    main()
