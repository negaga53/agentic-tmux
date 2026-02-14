"""Real-time monitoring dashboard for agentic sessions.

Displays a live TUI showing agent status, message activity, heartbeats,
and session health. Automatically launched in the admin pane.

Features:
- Arrow key navigation between agents/orchestrator
- Message details for selected entity
- Real-time activity log
- Send messages to agents/orchestrator
- Terminate all workers
- Interrupt and request status reports
- Detect when agents/orchestrator are not running

Usage:
    agentic monitor
    agentic-monitor
    
Controls:
    ↑/↓ or j/k: Navigate between agents
    m: Send message to selected agent
    t: Terminate all workers
    i: Interrupt all - request immediate reports
    e: Toggle expanded view
    q: Quit
"""

from __future__ import annotations

import fcntl
import json
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic.config import (
    get_activity_log,
    get_current_session_id,
    get_storage_client,
    ensure_config_dir,
    resolve_working_dir,
    WORKING_DIR_ENV_VAR,
)
from agentic.models import SessionStatus
from agentic.tmux_manager import TmuxManager


console = Console()


# =============================================================================
# Activity Log System
# =============================================================================

def log_activity(
    event_type: str, 
    details: dict[str, Any], 
    session_id: str | None = None,
    working_dir: str | None = None,
) -> None:
    """Log an activity event to the activity log file.
    
    Args:
        event_type: Type of event (session_start, agent_spawn, etc.)
        details: Event details dict
        session_id: Optional session ID
        working_dir: Working directory for per-repo storage. If None, resolved automatically.
    
    Event types:
        - session_start: Session started
        - session_stop: Session stopped
        - agent_spawn: Agent spawned
        - agent_terminate: Agent terminated
        - message_sent: Message sent between agents
        - message_received: Message received by agent
        - heartbeat: Heartbeat received (logged sparingly)
        - error: Error occurred
    """
    working_dir = resolve_working_dir(working_dir)
    
    config_dir = ensure_config_dir(working_dir)
    activity_log_file = get_activity_log(working_dir)
    
    entry = {
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S"),
        "event": event_type,
        "session_id": session_id,
        **details,
    }
    
    try:
        with open(activity_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Don't fail if logging fails


def read_activity_log(
    max_entries: int = 50, 
    session_id: str | None = None,
    working_dir: str | None = None,
) -> list[dict]:
    """Read recent entries from the activity log.
    
    Args:
        max_entries: Maximum number of entries to return
        session_id: Optional filter by session ID
        working_dir: Working directory. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    
    activity_log_file = get_activity_log(working_dir)
    if not activity_log_file.exists():
        return []
    
    entries = deque(maxlen=max_entries)
    try:
        with open(activity_log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Filter by session if specified
                    if session_id and entry.get("session_id") != session_id:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return list(entries)


def clear_activity_log(working_dir: str | None = None) -> None:
    """Clear the activity log file.
    
    Args:
        working_dir: Working directory. If None, resolved automatically.
    """
    working_dir = resolve_working_dir(working_dir)
    
    activity_log_file = get_activity_log(working_dir)
    if activity_log_file.exists():
        activity_log_file.unlink()


# =============================================================================
# Monitor State
# =============================================================================

@dataclass
class MonitorState:
    """State for the interactive monitor."""
    session_id: str
    working_dir: str = "."  # Working directory for per-repo storage
    selected_index: int = 0  # 0 = orchestrator, 1+ = agents
    running: bool = True
    session_ended: bool = False
    end_reason: str = ""
    expanded_view: bool = False  # Show full messages without truncation
    # Input mode state
    input_mode: bool = False  # True when collecting user input
    input_prompt: str = ""  # Prompt to display
    input_buffer: str = ""  # Currently typed text
    input_callback: str = ""  # Action to perform on Enter (send_message, etc.)
    # Status message (shown briefly)
    status_message: str = ""
    status_time: float = 0.0


# =============================================================================
# Keyboard Input
# =============================================================================

def get_key_nonblocking() -> str | None:
    """Get a keypress without blocking. Returns None if no key pressed."""
    # Check if data available
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    if not rlist:
        return None
    
    ch = sys.stdin.read(1)
    if not ch:
        return None
    
    # Handle escape sequences (arrow keys)
    if ch == '\x1b':
        # Try to read more - arrow keys send ESC [ A/B/C/D
        # Set non-blocking temporarily to read rest of sequence
        fd = sys.stdin.fileno()
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        
        try:
            # Read up to 4 more chars for escape sequence
            rest = ''
            for _ in range(4):
                try:
                    c = sys.stdin.read(1)
                    if c:
                        rest += c
                    else:
                        break
                except (IOError, BlockingIOError):
                    break
            
            buf = ch + rest
            
            # Parse escape sequences
            if buf == '\x1b[A' or buf == '\x1bOA':
                return 'UP'
            elif buf == '\x1b[B' or buf == '\x1bOB':
                return 'DOWN'
            elif buf == '\x1b[C' or buf == '\x1bOC':
                return 'RIGHT'
            elif buf == '\x1b[D' or buf == '\x1bOD':
                return 'LEFT'
            return 'ESC'
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
    
    return ch


# =============================================================================
# Helper Functions
# =============================================================================


def truncate_text(text: str, max_len: int = 40) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def format_timestamp(ts: float) -> str:
    """Format a timestamp as HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.localtime(ts))


# =============================================================================
# Agent Liveness Detection
# =============================================================================

def check_agent_liveness(storage: Any, session_id: str) -> dict[str, dict[str, Any]]:
    """Check if agents' tmux panes are still running.
    
    Returns a dict mapping agent_id to liveness info:
        {
            "agent_id": {
                "pane_exists": bool,
                "is_running": bool,  # pane exists and has an active command
                "current_command": str | None,
            }
        }
    """
    agents = storage.get_all_agents(session_id)
    liveness = {}
    
    try:
        tmux = TmuxManager()
        if not tmux.session_exists():
            # Session gone, all agents dead
            for agent in agents:
                liveness[agent.id] = {
                    "pane_exists": False,
                    "is_running": False,
                    "current_command": None,
                }
            return liveness
        
        # Get all panes in the session
        all_panes = tmux.list_all_panes()
        pane_lookup = {p.pane_id: p for p in all_panes}
        
        for agent in agents:
            pane_id = agent.pane_id
            if not pane_id or pane_id not in pane_lookup:
                liveness[agent.id] = {
                    "pane_exists": False,
                    "is_running": False,
                    "current_command": None,
                }
            else:
                pane_info = pane_lookup[pane_id]
                cmd = pane_info.current_command or ""
                # Check if command indicates active work (copilot, claude, etc.)
                # If the pane shows bash/zsh/sh, the CLI likely exited
                is_active_cli = any(cli in cmd.lower() for cli in ["copilot", "claude", "aider", "node", "python"])
                liveness[agent.id] = {
                    "pane_exists": True,
                    "is_running": is_active_cli,
                    "current_command": cmd,
                }
    except Exception:
        # Tmux not available or error
        for agent in agents:
            liveness[agent.id] = {
                "pane_exists": None,  # Unknown
                "is_running": None,
                "current_command": None,
            }
    
    return liveness


def check_orchestrator_liveness(working_dir: str) -> dict[str, Any]:
    """Check if the orchestrator process is running.
    
    Returns:
        {
            "pid": int | None,
            "is_running": bool,
        }
    """
    from agentic.config import get_pid_file
    
    pid_file = get_pid_file(working_dir)
    if not pid_file.exists():
        return {"pid": None, "is_running": False}
    
    try:
        pid = int(pid_file.read_text().strip())
        # Check if process exists
        import os
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        return {"pid": pid, "is_running": True}
    except (ValueError, ProcessLookupError, PermissionError):
        return {"pid": None, "is_running": False}


# =============================================================================
# Admin Commands
# =============================================================================

def send_message_to_entity(
    storage: Any,
    session_id: str,
    entity_id: str,
    message: str,
    working_dir: str,
) -> bool:
    """Send a message to an agent or orchestrator.
    
    Args:
        storage: Storage client
        session_id: Session ID
        entity_id: Target agent ID or "orchestrator"
        message: Message to send
        working_dir: Working directory for logging
    
    Returns:
        True if successful.
    """
    try:
        msg_id = storage.send_agent_message(
            session_id=session_id,
            from_agent="admin",
            to_agent=entity_id,
            message=message,
        )
        
        # Log message sent
        log_activity("message_sent", {
            "from": "admin",
            "to": entity_id,
            "message_preview": message[:500],
        }, session_id=session_id, working_dir=working_dir)
        
        return True
    except Exception:
        return False


def terminate_all_workers(storage: Any, session_id: str, working_dir: str) -> int:
    """Send TERMINATE to all agents and mark session as done.
    
    Returns:
        Number of agents terminated.
    """
    agents = storage.get_all_agents(session_id)
    
    terminated = 0
    for agent in agents:
        try:
            storage.send_agent_message(
                session_id=session_id,
                from_agent="admin",
                to_agent=agent.id,
                message="TERMINATE",
            )
            terminated += 1
            
            # Log agent terminate
            log_activity("agent_terminate", {
                "agent_id": agent.id,
            }, session_id=session_id, working_dir=working_dir)
        except Exception:
            pass
    
    # Mark session as done
    storage.push_done_to_all(session_id)
    
    return terminated


def interrupt_and_report(storage: Any, session_id: str, working_dir: str) -> int:
    """Send interrupt signal to all workers telling them to report and exit.
    
    This sends a special message instructing agents to:
    1. Stop what they're doing
    2. Send their current progress/results to the orchestrator
    3. Exit gracefully
    
    Returns:
        Number of agents messaged.
    """
    agents = storage.get_all_agents(session_id)
    
    interrupt_message = """INTERRUPT: Session admin has requested an immediate status report.

REQUIRED ACTIONS:
1. STOP any current work immediately
2. Compile your current progress, partial results, or status
3. Send a report to orchestrator using: send_to_agent(agent_id="orchestrator", message="<your progress report>")
4. After sending the report, exit gracefully

This is an administrative interrupt. Comply immediately."""
    
    messaged = 0
    for agent in agents:
        try:
            storage.send_agent_message(
                session_id=session_id,
                from_agent="admin",
                to_agent=agent.id,
                message=interrupt_message,
            )
            messaged += 1
            
            # Log the interrupt
            log_activity("message_sent", {
                "from": "admin",
                "to": agent.id,
                "message_preview": "INTERRUPT: Request immediate status report",
            }, session_id=session_id, working_dir=working_dir)
        except Exception:
            pass
    
    return messaged


# =============================================================================
# Panel Builders
# =============================================================================

def build_header_panel(state: MonitorState, session: Any, start_time: float) -> Panel:
    """Build the session header panel."""
    uptime = int(time.time() - start_time)
    uptime_str = f"{uptime // 60}m {uptime % 60}s"
    
    if state.session_ended:
        status_style = "red" if state.end_reason == "failed" else "yellow"
        status_text = f"ENDED ({state.end_reason})"
    elif session.status == SessionStatus.RUNNING:
        status_style = "green"
        status_text = session.status.value
    else:
        status_style = "yellow"
        status_text = session.status.value
    
    header_text = Text()
    header_text.append("Session: ", style="bold")
    header_text.append(f"{state.session_id}  ", style="cyan")
    header_text.append("Status: ", style="bold")
    header_text.append(f"{status_text}  ", style=status_style)
    header_text.append("Uptime: ", style="bold")
    header_text.append(f"{uptime_str}  ", style="dim")
    
    # Show status message if recent (within 3 seconds)
    if state.status_message and (time.time() - state.status_time) < 3.0:
        header_text.append(f"\n{state.status_message}", style="yellow bold")
    elif state.input_mode:
        header_text.append(f"\n{state.input_prompt}: ", style="yellow bold")
        header_text.append(state.input_buffer, style="white")
        header_text.append("█", style="white blink")
    else:
        expand_hint = "[e] Collapse" if state.expanded_view else "[e] Expand"
        header_text.append(f"\n[↑↓] Navigate  [m] Message  [t] Terminate  [i] Interrupt  {expand_hint}  [q] Quit", style="dim italic")
    
    return Panel(header_text, title="[bold blue]AGENTIC MONITOR[/bold blue]", border_style="blue", height=4)


def build_entities_table(storage: Any, state: MonitorState) -> Table:
    """Build the entities (orchestrator + agents) table with selection highlight."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold")
    table.add_column("", width=2)  # Selection indicator
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Role", no_wrap=True, overflow="ellipsis", ratio=1)  # Single line, truncate with ellipsis, use remaining space
    table.add_column("Status", width=6)
    table.add_column("HB", justify="right", width=4)
    
    agents = storage.get_all_agents(state.session_id)
    current_time = time.time()
    
    # Orchestrator row (index 0)
    orch_selected = state.selected_index == 0
    orch_indicator = "►" if orch_selected else " "
    orch_style = "reverse" if orch_selected else ""
    
    table.add_row(
        Text(orch_indicator, style="green bold"),
        Text("orch", style=f"bold cyan {orch_style}"),
        Text("Coordinator", style=f"dim {orch_style}"),
        Text("-", style=orch_style),
        Text("-", style=orch_style),
    )
    
    # Agent rows
    for i, agent in enumerate(sorted(agents, key=lambda a: a.id)):
        agent_index = i + 1  # +1 because orchestrator is 0
        selected = state.selected_index == agent_index
        indicator = "►" if selected else " "
        row_style = "reverse" if selected else ""
        
        # Heartbeat formatting (skip stale warnings if session ended)
        heartbeat_age = current_time - agent.last_heartbeat
        if state.session_ended:
            hb_text = Text("-", style=row_style)
        elif heartbeat_age < 30:
            hb_text = Text(f"{int(heartbeat_age)}s", style=f"green {row_style}")
        elif heartbeat_age < 60:
            hb_text = Text(f"{int(heartbeat_age)}s", style=f"yellow {row_style}")
        else:
            hb_text = Text(f"{int(heartbeat_age//60)}m", style=f"red {row_style}")
        
        # Status formatting
        status = agent.status.value
        if status == "working":
            status_text = Text("work", style=f"green {row_style}")
        elif status == "idle":
            status_text = Text("idle", style=f"dim {row_style}")
        elif status == "polling":
            status_text = Text("poll", style=f"cyan {row_style}")
        elif status == "waiting":
            status_text = Text("wait", style=f"yellow {row_style}")
        elif status == "failed":
            status_text = Text("fail", style=f"red {row_style}")
        elif status == "done":
            status_text = Text("done", style=f"blue {row_style}")
        else:
            status_text = Text(status[:4], style=row_style)
        
        table.add_row(
            Text(indicator, style="green bold"),
            Text(agent.id, style=f"cyan {row_style}"),
            Text(agent.role, style=row_style),  # Full role, no truncation
            status_text,
            hb_text,
        )
    
    if not agents:
        table.add_row("", "", Text("No agents spawned yet", style="dim"), "", "")
    
    return table


def get_message_history(session_id: str, entity_id: str, max_entries: int = 20, working_dir: str | None = None) -> list[dict]:
    """Get message history (sent and received) for an entity from activity log.
    
    Only uses message_sent events to avoid duplicates (message_received would double-count).
    """
    entries = read_activity_log(max_entries=200, session_id=session_id, working_dir=working_dir)
    history = []
    
    for entry in entries:
        event = entry.get("event", "")
        # Only use message_sent to avoid duplicates with message_received
        if event == "message_sent":
            if entry.get("from") == entity_id or entry.get("to") == entity_id:
                # Clean preview: collapse whitespace and newlines
                preview = entry.get("message_preview", "")
                preview = " ".join(preview.split())
                history.append({
                    "type": "sent" if entry.get("from") == entity_id else "received",
                    "from": entry.get("from", "?"),
                    "to": entry.get("to", "?"),
                    "preview": preview,
                    "time": entry.get("time_str", "")
                })
    
    return history[-max_entries:]


def build_messages_panel(storage: Any, state: MonitorState) -> Panel:
    """Build message details panel for selected entity."""
    agents = storage.get_all_agents(state.session_id)
    expanded = state.expanded_view
    working_dir = state.working_dir
    
    # Determine selected entity
    selected_agent = None
    if state.selected_index == 0:
        entity_id = "orchestrator"
        entity_name = "Orchestrator"
        entity_role = "Session coordinator"
        entity_status = "-"
    else:
        sorted_agents = sorted(agents, key=lambda a: a.id)
        agent_index = state.selected_index - 1
        if agent_index < len(sorted_agents):
            selected_agent = sorted_agents[agent_index]
            entity_id = selected_agent.id
            entity_name = f"Agent {entity_id}"
            entity_role = selected_agent.role
            entity_status = selected_agent.status.value
        else:
            entity_id = "orchestrator"
            entity_name = "Orchestrator"
            entity_role = "Session coordinator"
            entity_status = "-"
    
    # Get pending messages in queue
    messages = storage.peek_agent_messages(state.session_id, entity_id, count=5)
    msg_count = storage.get_message_count(state.session_id, entity_id)
    
    # Get message history from activity log
    history = get_message_history(state.session_id, entity_id, max_entries=10 if not expanded else 20, working_dir=working_dir)
    
    content = Text()
    content.append(f"{entity_name}", style="cyan bold")
    content.append(f" [{entity_status}]", style="green" if entity_status == "working" else "dim")
    content.append(f" - {msg_count} pending\n", style="dim")
    content.append(f"Role: ", style="bold")
    content.append(f"{entity_role}\n", style="white")  # Full role, no truncation
    content.append("─" * 40 + "\n", style="dim")
    
    # Show pending queue
    if messages:
        content.append("📥 QUEUE: ", style="yellow bold")
        for msg in messages[:2]:
            from_agent = msg.get("from", "?")
            content.append(f"←{from_agent} ", style="cyan")
        content.append("\n")
    
    # Show history (compact)
    if not history:
        content.append("No message history\n", style="dim")
    else:
        for item in history[-8:]:
            time_str = item.get("time", "")
            msg_type = item.get("type", "")
            preview = item.get("preview", "")[:60]
            if len(item.get("preview", "")) > 60:
                preview += "..."
            
            if msg_type == "sent":
                to_agent = item.get("to", "?")
                content.append(f"{time_str} ", style="dim")
                content.append(f"→{to_agent} ", style="green")
                content.append(f"{preview}\n", style="white")
            else:
                from_agent = item.get("from", "?")
                content.append(f"{time_str} ", style="dim")
                content.append(f"←{from_agent} ", style="blue")
                content.append(f"{preview}\n", style="white")
    
    return Panel(content, title=f"[bold green]{entity_name}[/bold green]", border_style="green")


def build_activity_log_panel(session_id: str, max_lines: int = 50, expanded: bool = False, working_dir: str | None = None) -> Panel:
    """Build the activity log panel showing recent events."""
    entries = read_activity_log(max_entries=max_lines, session_id=session_id, working_dir=working_dir)
    
    content = Text()
    
    if not entries:
        content.append("No activity logged yet\n", style="dim")
        content.append("Activity will appear as agents communicate", style="dim")
    else:
        # Show entries
        for entry in entries[-max_lines:]:
            time_str = entry.get("time_str", "??:??:??")
            event = entry.get("event", "unknown")
            
            # Format based on event type
            if event == "session_start":
                content.append(f"{time_str} ", style="dim")
                content.append("START ", style="green bold")
                content.append("Session started\n", style="green")
            elif event == "session_stop":
                content.append(f"{time_str} ", style="dim")
                content.append("STOP  ", style="red bold")
                content.append("Session stopped\n", style="red")
            elif event == "agent_spawn":
                agent_id = entry.get("agent_id", "?")
                raw_role = entry.get("role", "")
                role = raw_role if expanded else truncate_text(raw_role, 40)
                content.append(f"{time_str} ", style="dim")
                content.append("SPAWN ", style="cyan bold")
                content.append(f"{agent_id}", style="cyan")
                if role:
                    content.append(f" ({role})", style="dim")
                content.append("\n")
            elif event == "agent_terminate":
                agent_id = entry.get("agent_id", "?")
                content.append(f"{time_str} ", style="dim")
                content.append("TERM  ", style="yellow bold")
                content.append(f"{agent_id}\n", style="yellow")
            elif event == "message_sent":
                from_agent = entry.get("from", "?")
                to_agent = entry.get("to", "?")
                raw_preview = entry.get("message_preview", "")
                # Collapse newlines but don't truncate - let panel wrap
                preview = " ".join(raw_preview.split())
                content.append(f"{time_str} ", style="dim")
                content.append("MSG   ", style="magenta bold")
                content.append(f"{from_agent}→{to_agent}", style="magenta")
                if preview:
                    content.append(f" {preview}", style="white")
                content.append("\n")
            elif event == "message_received":
                agent_id = entry.get("agent_id", "?")
                from_agent = entry.get("from", "?")
                content.append(f"{time_str} ", style="dim")
                content.append("RECV  ", style="blue bold")
                content.append(f"{agent_id}←{from_agent}\n", style="blue")
            elif event == "polling_start":
                agent_id = entry.get("agent_id", "?")
                timeout = entry.get("timeout", "?")
                content.append(f"{time_str} ", style="dim")
                content.append("POLL  ", style="cyan bold")
                content.append(f"{agent_id}", style="cyan")
                content.append(f" waiting {timeout}s\n", style="dim")
            elif event == "error":
                raw_msg = entry.get("message", "unknown error")
                msg = raw_msg if expanded else truncate_text(raw_msg, 60)
                content.append(f"{time_str} ", style="dim")
                content.append("ERROR ", style="red bold")
                content.append(f"{msg}\n", style="red")
            else:
                content.append(f"{time_str} ", style="dim")
                content.append(f"{event.upper():6}", style="white bold")
                content.append("\n")
    
    title = "[bold magenta]Activity Log[/bold magenta]"
    if expanded:
        title += " [EXPANDED]"
    return Panel(content, title=title, border_style="magenta")


def build_health_panel(storage: Any, state: MonitorState) -> Panel:
    """Build the system health panel."""
    issues = []
    agents = storage.get_all_agents(state.session_id)
    current_time = time.time()
    
    # Check agent liveness via tmux
    liveness = check_agent_liveness(storage, state.session_id)
    
    # Check orchestrator liveness
    orch_liveness = check_orchestrator_liveness(state.working_dir)
    
    # Only check health if session is running
    if not state.session_ended:
        # Check orchestrator
        if not orch_liveness["is_running"]:
            issues.append(Text("✖ Orchestrator not running", style="red bold"))
        
        # Check for stale heartbeats
        for agent in agents:
            age = current_time - agent.last_heartbeat
            if age > 60:
                issues.append(Text(f"⚠ {agent.id} stale ({int(age)}s)", style="red"))
        
        # Check for queue backlogs
        for agent in agents:
            if agent.task_queue_length > 5:
                issues.append(Text(f"⚠ {agent.id} backlog ({agent.task_queue_length})", style="yellow"))
        
        # Check for dead agent panes
        for agent_id, info in liveness.items():
            if info["pane_exists"] is False:
                issues.append(Text(f"✖ {agent_id} pane gone", style="red"))
            elif info["is_running"] is False and info["pane_exists"]:
                issues.append(Text(f"⚠ {agent_id} CLI exited", style="yellow"))
    
    # Session status
    session = storage.get_session(state.session_id)
    if state.session_ended:
        if state.end_reason == "failed":
            issues.append(Text("✖ Session FAILED", style="red bold"))
        else:
            issues.append(Text("● Session ended", style="yellow"))
    elif session and session.status == SessionStatus.FAILED:
        issues.append(Text("✖ Session FAILED", style="red bold"))
    
    if not issues:
        content = Text("✓ All healthy", style="green bold")
    else:
        content = Text()
        for issue in issues:
            content.append(issue)
            content.append("\n")
    
    return Panel(content, title="[bold]Health[/bold]", border_style="cyan")


def build_dashboard(storage: Any, state: MonitorState, start_time: float) -> Layout:
    """Build the complete dashboard layout.
    
    Layout:
    ┌─────────────────────────────────────┬──────────┐
    │             Agents                  │  Health  │
    ├──────────────────┬──────────────────┴──────────┤
    │                  │                             │
    │  Activity Logs   │         Messages            │
    │                  │                             │
    └──────────────────┴─────────────────────────────┘
    """
    session = storage.get_session(state.session_id)
    if not session:
        return Layout(Panel("[red]Session not found[/red]", title="Error"))
    
    layout = Layout()
    
    # Main vertical split: header, top row (agents+health), bottom row (activity+messages)
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="top", size=10),
        Layout(name="bottom"),
    )
    
    layout["header"].update(build_header_panel(state, session, start_time))
    
    # Top row: Agents (big) | Health (small)
    layout["top"].split_row(
        Layout(name="agents", ratio=4),
        Layout(name="health", ratio=1),
    )
    
    entities_panel = Panel(
        build_entities_table(storage, state),
        title="[bold cyan]Agents[/bold cyan]",
        border_style="cyan"
    )
    layout["top"]["agents"].update(entities_panel)
    layout["top"]["health"].update(build_health_panel(storage, state))
    
    # Bottom row: Activity Logs (left) | Messages (right)
    layout["bottom"].split_row(
        Layout(name="activity", ratio=1),
        Layout(name="messages", ratio=1),
    )
    
    layout["bottom"]["activity"].update(build_activity_log_panel(state.session_id, max_lines=25 if not state.expanded_view else 40, expanded=state.expanded_view, working_dir=state.working_dir))
    layout["bottom"]["messages"].update(build_messages_panel(storage, state))
    
    return layout


# =============================================================================
# Main Monitor Loop
# =============================================================================

def wait_for_session(timeout: int = 300, working_dir: str | None = None) -> tuple[str | None, str]:
    """Wait for a session to become available.
    
    Args:
        timeout: Seconds to wait for session
        working_dir: Working directory. If None, uses CWD.
    
    Returns:
        Tuple of (session_id, working_dir) or (None, working_dir) if timeout.
    """
    if working_dir is None:
        working_dir = os.getcwd()
    
    console.print("[dim]Waiting for agentic session to start...[/dim]")
    console.print("[dim]Press Ctrl+C to cancel[/dim]")
    
    start = time.time()
    while time.time() - start < timeout:
        session_id = get_current_session_id(working_dir)
        if session_id:
            storage = get_storage_client(working_dir)
            session = storage.get_session(session_id)
            if session:
                return session_id, session.working_directory
        time.sleep(1)
    
    return None, working_dir


def run_monitor(refresh_rate: float = 1.0) -> None:
    """Run the interactive monitoring dashboard."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    
    if not session_id:
        session_id, working_dir = wait_for_session(working_dir=working_dir)
        if not session_id:
            console.print("[red]Timeout waiting for session[/red]")
            sys.exit(1)
    else:
        # Get actual working_dir from session
        storage = get_storage_client(working_dir)
        session = storage.get_session(session_id)
        if session:
            working_dir = session.working_directory
    
    storage = get_storage_client(working_dir)
    start_time = time.time()
    
    state = MonitorState(session_id=session_id, working_dir=working_dir)
    
    # Save terminal settings and set to cbreak mode for key capture
    # (cbreak is less aggressive than raw and works better with Rich)
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
    except termios.error:
        pass  # Not a TTY, skip keyboard input
    
    try:
        if old_settings:
            tty.setcbreak(sys.stdin.fileno())
        
        with Live(console=console, refresh_per_second=4, screen=True, transient=False) as live:
            last_data_update = 0
            needs_render = True  # Force initial render
            
            while state.running:
                # Handle keyboard input (only if TTY) - check frequently for responsiveness
                if old_settings:
                    key = get_key_nonblocking()
                    if key:
                        agents = storage.get_all_agents(state.session_id)
                        max_index = len(agents)  # 0 = orchestrator, 1..n = agents
                        
                        # Input mode handling
                        if state.input_mode:
                            if key == '\x1b' or key == 'ESC':  # Escape - cancel input
                                state.input_mode = False
                                state.input_buffer = ""
                                state.input_callback = ""
                                needs_render = True
                            elif key == '\r' or key == '\n':  # Enter - submit input
                                if state.input_buffer.strip():
                                    # Execute the callback action
                                    if state.input_callback == "send_message":
                                        # Determine target entity
                                        if state.selected_index == 0:
                                            target_id = "orchestrator"
                                        else:
                                            sorted_agents = sorted(agents, key=lambda a: a.id)
                                            agent_index = state.selected_index - 1
                                            if agent_index < len(sorted_agents):
                                                target_id = sorted_agents[agent_index].id
                                            else:
                                                target_id = "orchestrator"
                                        
                                        if send_message_to_entity(storage, state.session_id, target_id, state.input_buffer, state.working_dir):
                                            state.status_message = f"✓ Sent to {target_id}"
                                            state.status_time = time.time()
                                        else:
                                            state.status_message = f"✖ Failed to send to {target_id}"
                                            state.status_time = time.time()
                                
                                state.input_mode = False
                                state.input_buffer = ""
                                state.input_callback = ""
                                needs_render = True
                            elif key == '\x7f' or key == '\b':  # Backspace
                                if state.input_buffer:
                                    state.input_buffer = state.input_buffer[:-1]
                                    needs_render = True
                            elif len(key) == 1 and key.isprintable():  # Regular character
                                state.input_buffer += key
                                needs_render = True
                            continue
                        
                        # Normal mode key handling
                        if key in ('q', 'Q', '\x03'):  # q or Ctrl+C
                            state.running = False
                            continue
                        elif key in ('UP', 'k'):
                            new_index = max(0, state.selected_index - 1)
                            if new_index != state.selected_index:
                                state.selected_index = new_index
                                needs_render = True
                        elif key in ('DOWN', 'j'):
                            new_index = min(max_index, state.selected_index + 1)
                            if new_index != state.selected_index:
                                state.selected_index = new_index
                                needs_render = True
                        elif key in ('e', 'E'):
                            state.expanded_view = not state.expanded_view
                            needs_render = True
                        elif key in ('m', 'M'):
                            # Send message to selected entity
                            if state.selected_index == 0:
                                target = "orchestrator"
                            else:
                                sorted_agents = sorted(agents, key=lambda a: a.id)
                                agent_index = state.selected_index - 1
                                if agent_index < len(sorted_agents):
                                    target = sorted_agents[agent_index].id
                                else:
                                    target = "orchestrator"
                            state.input_mode = True
                            state.input_prompt = f"Message to {target} (ESC=cancel)"
                            state.input_buffer = ""
                            state.input_callback = "send_message"
                            needs_render = True
                        elif key in ('t', 'T'):
                            # Terminate all workers
                            count = terminate_all_workers(storage, state.session_id, state.working_dir)
                            state.status_message = f"✓ Terminated {count} agents"
                            state.status_time = time.time()
                            needs_render = True
                        elif key in ('i', 'I'):
                            # Interrupt and request reports
                            count = interrupt_and_report(storage, state.session_id, state.working_dir)
                            state.status_message = f"✓ Interrupt sent to {count} agents"
                            state.status_time = time.time()
                            needs_render = True
                
                # Check session status and refresh data periodically
                now = time.time()
                if now - last_data_update >= refresh_rate:
                    session = storage.get_session(state.session_id)
                    if not session:
                        state.session_ended = True
                        state.end_reason = "deleted"
                    elif session.status == SessionStatus.COMPLETED:
                        state.session_ended = True
                        state.end_reason = "completed"
                    elif session.status == SessionStatus.FAILED:
                        state.session_ended = True
                        state.end_reason = "failed"
                    
                    last_data_update = now
                    needs_render = True
                
                # Render when needed (keypress or data refresh)
                if needs_render:
                    live.update(build_dashboard(storage, state, start_time))
                    needs_render = False
                
                # Very small sleep - just enough to not spin CPU
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass
        console.print("\n[dim]Monitor stopped[/dim]")


def main():
    """Entry point for the monitor."""
    run_monitor()


if __name__ == "__main__":
    main()
