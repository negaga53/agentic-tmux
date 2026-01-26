"""CLI interface for agentic-tmux."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from agentic.dag import create_task_flow_diagram, validate_dag
from agentic.models import (
    Agent,
    AgenticSession,
    ExecutionPlan,
    SessionStatus,
    Task,
    TaskStatus,
)
from agentic.orchestrator import start_orchestrator_background, stop_orchestrator
from agentic.planner import get_planner
from agentic.redis_client import AgenticRedisClient
from agentic.tmux_manager import TmuxManager, check_tmux_available

console = Console()

# Config paths
CONFIG_DIR = Path.home() / ".config" / "agentic"
PID_FILE = CONFIG_DIR / "orchestrator.pid"
SESSION_FILE = CONFIG_DIR / "current_session"


def get_redis_client() -> AgenticRedisClient:
    """Get Redis client with config from environment."""
    return AgenticRedisClient(
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


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Agentic TMUX - Multi-agent orchestration for CLI coding assistants."""
    pass


@main.command()
@click.option("--working-dir", "-d", default=".", help="Working directory")
@click.option("--cli", default="gh copilot", help="CLI command to use (e.g., 'gh copilot', 'claude')")
def start(working_dir: str, cli: str):
    """Start a new agentic session."""
    # Check prerequisites
    if not check_tmux_available():
        console.print("[red]Error:[/red] tmux is not installed or not in PATH")
        sys.exit(1)
    
    redis = get_redis_client()
    if not redis.ping():
        console.print("[red]Error:[/red] Cannot connect to Redis. Please start Redis first.")
        console.print("  Run: [cyan]redis-server[/cyan]")
        sys.exit(1)
    
    # Check for existing session
    existing_id = get_current_session_id()
    if existing_id:
        session = redis.get_session(existing_id)
        if session and session.status not in (SessionStatus.COMPLETED, SessionStatus.FAILED):
            console.print(f"[yellow]Warning:[/yellow] Session {existing_id} already exists.")
            if not Confirm.ask("Start a new session anyway?"):
                sys.exit(0)
    
    # Create new session
    working_dir = os.path.abspath(working_dir)
    session = AgenticSession(working_directory=working_dir)
    session.config["cli_command"] = cli
    
    redis.create_session(session)
    save_current_session_id(session.id)
    
    # Create tmux session with admin pane
    tmux = TmuxManager()
    admin_pane_id = tmux.create_admin_pane(working_dir)
    session.admin_pane_id = admin_pane_id
    redis.update_session_status(session.id, SessionStatus.RUNNING)
    
    # Start orchestrator daemon
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG_DIR / f"orchestrator_{session.id}.log"
    pid = start_orchestrator_background(
        session_id=session.id,
        log_file=str(log_file),
    )
    
    if pid:
        PID_FILE.write_text(str(pid))
        console.print(f"[green]✓[/green] Started orchestrator (PID: {pid})")
    
    console.print(f"\n[green]✓[/green] Session [cyan]{session.id}[/cyan] started")
    console.print(f"  Working directory: {working_dir}")
    console.print(f"  CLI: {cli}")
    console.print(f"\n  To attach: [cyan]tmux attach -t agentic[/cyan]")
    console.print(f"  To create plan: [cyan]agentic plan \"your task\"[/cyan]")


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
@click.argument("prompt")
@click.option("--agents", "-a", default=3, help="Suggested number of agents")
@click.option("--no-llm", is_flag=True, help="Use simple rule-based planning")
def plan(prompt: str, agents: int, no_llm: bool):
    """Create an execution plan for a task."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[red]Error:[/red] No active session. Run [cyan]agentic start[/cyan] first.")
        sys.exit(1)
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    if not session:
        console.print("[red]Error:[/red] Session not found")
        sys.exit(1)
    
    # Get file context
    working_dir = session.working_directory
    try:
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.ts", "-o", "-name", "*.js"],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=5,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()][:100]
    except Exception:
        files = []
    
    # Create plan
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Creating execution plan...", total=None)
        
        planner = get_planner(use_llm=not no_llm)
        execution_plan = planner.create_plan(
            prompt=prompt,
            working_dir=working_dir,
            suggested_agents=agents,
            file_context=files,
        )
    
    # Display plan
    display_plan(execution_plan)
    
    # Ask for approval
    console.print()
    choice = Prompt.ask(
        "Options: [a]pprove, [m]odify, [+]add agent, [-]remove agent, [c]ancel",
        choices=["a", "m", "+", "-", "c"],
        default="a",
    )
    
    if choice == "c":
        console.print("[yellow]Plan cancelled[/yellow]")
        return
    
    if choice == "m":
        feedback = Prompt.ask("Enter modifications")
        planner = get_planner(use_llm=not no_llm)
        if hasattr(planner, "refine_plan"):
            execution_plan = planner.refine_plan(execution_plan, feedback)
            display_plan(execution_plan)
            if not Confirm.ask("Approve modified plan?"):
                return
    
    if choice == "+":
        role = Prompt.ask("New agent role")
        scope = Prompt.ask("File scope patterns (comma-separated)", default="**/*")
        new_agent = Agent(
            role=role,
            scope={"patterns": [s.strip() for s in scope.split(",")]},
        )
        execution_plan.agents.append(new_agent)
        display_plan(execution_plan)
        if not Confirm.ask("Approve modified plan?"):
            return
    
    # Validate DAG
    valid, errors = validate_dag(execution_plan.dag)
    if not valid:
        console.print("[red]Plan validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        return
    
    # Execute plan
    execute_plan(session_id, execution_plan, redis)


def display_plan(plan: ExecutionPlan) -> None:
    """Display an execution plan with rich formatting."""
    # Header
    console.print()
    console.print(Panel(
        f"[bold]Prompt:[/bold] {plan.prompt}",
        title="📋 EXECUTION PLAN",
        border_style="blue",
    ))
    
    # Agents table
    agent_table = Table(title=f"AGENTS TO SPAWN: {len(plan.agents)}")
    agent_table.add_column("ID", style="cyan")
    agent_table.add_column("Role", style="green")
    agent_table.add_column("File Scope")
    
    for agent in plan.agents:
        scope_str = ", ".join(agent.scope.patterns[:3])
        if len(agent.scope.patterns) > 3:
            scope_str += f" (+{len(agent.scope.patterns) - 3} more)"
        if agent.scope.read_only:
            scope_str = f"READ-ONLY: {scope_str}"
        agent_table.add_row(agent.id, agent.role, scope_str)
    
    console.print(agent_table)
    
    # Task flow
    console.print("\n[bold]TASK FLOW:[/bold]")
    console.print(create_task_flow_diagram(plan.dag))
    
    # Communications
    if plan.estimated_communications:
        console.print("\n[bold]ESTIMATED COMMUNICATIONS:[/bold]")
        for comm in plan.estimated_communications:
            console.print(f"  • {comm}")


def execute_plan(session_id: str, plan: ExecutionPlan, redis: AgenticRedisClient) -> None:
    """Execute an approved plan by spawning agents and dispatching tasks."""
    session = redis.get_session(session_id)
    if not session:
        console.print("[red]Error:[/red] Session not found")
        return
    
    tmux = TmuxManager()
    cli_command = session.config.get("cli_command", "gh copilot")
    
    # Spawn worker panes
    console.print("\n[bold]Spawning agents...[/bold]")
    
    pane_mapping = tmux.spawn_multiple_workers(
        agents=plan.agents,
        working_dir=session.working_directory,
        cli_command=cli_command,
        session_id=session_id,
    )
    
    # Register agents in Redis
    for agent in plan.agents:
        agent.pane_id = pane_mapping.get(agent.id)
        redis.register_agent(session_id, agent)
        console.print(f"  [green]✓[/green] {agent.id}: {agent.role} (pane {agent.pane_id})")
    
    # Save DAG
    redis.save_dag(session_id, plan.dag)
    
    # Dispatch initial tasks
    console.print("\n[bold]Dispatching tasks...[/bold]")
    
    ready_tasks = plan.dag.get_ready_tasks()
    for task in ready_tasks:
        if task.agent_id:
            redis.push_task(session_id, task.agent_id, task)
            redis.update_task_status(session_id, task.id, TaskStatus.PENDING)
            console.print(f"  [green]✓[/green] {task.id} → {task.agent_id}: {task.title}")
    
    console.print(f"\n[green]✓[/green] Plan execution started")
    console.print(f"  Monitor with: [cyan]agentic status[/cyan]")
    console.print(f"  View logs: [cyan]agentic logs <agent_id>[/cyan]")


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


def render_status(session_id: str, redis: AgenticRedisClient) -> Panel:
    """Render status panel."""
    session = redis.get_session(session_id)
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
    
    agents = redis.get_all_agents(session_id)
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
    dag = redis.get_dag(session_id)
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
    logs = redis.get_agent_logs(session_id, agent_id, count=lines)
    
    for log in logs:
        ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
        file_str = f" ({log.file})" if log.file else ""
        console.print(f"[dim]{ts}[/dim] [{log.agent_id}] {log.action}{file_str}")
    
    if follow:
        last_id = "0" if not logs else str(int(logs[-1].timestamp * 1000))
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
        from_agent="admin",
    )
    
    redis.push_task(session_id, agent_id, task)
    console.print(f"[green]✓[/green] Task sent to {agent_id}")


@main.command()
def resume():
    """Resume session with existing agents (for new prompts)."""
    session_id = get_current_session_id()
    if not session_id:
        console.print("[red]Error:[/red] No active session")
        sys.exit(1)
    
    redis = get_redis_client()
    session = redis.get_session(session_id)
    
    if not session:
        console.print("[red]Error:[/red] Session not found")
        sys.exit(1)
    
    agents = redis.get_all_agents(session_id)
    if not agents:
        console.print("[yellow]No agents in session. Use [cyan]agentic plan[/cyan] to create a new plan.[/yellow]")
        return
    
    console.print(f"Session [cyan]{session_id}[/cyan] has {len(agents)} agents:")
    for agent in agents:
        console.print(f"  • {agent.id}: {agent.role}")
    
    console.print("\nUse [cyan]agentic plan \"your task\"[/cyan] to assign new work")


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
    
    # Clear DAG
    redis.save_dag(session_id, TaskDAG())
    
    console.print("[green]✓[/green] Session cleared, ready for new plan")


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
        logs = redis.get_agent_logs(session_id, agent.id, count=1000)
        export_data["logs"][agent.id] = [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "file": log.file,
                "tool": log.tool,
            }
            for log in logs
        ]
    
    dag = redis.get_dag(session_id)
    if dag:
        export_data["dag"] = dag.to_dict()
    
    # Write to file
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)
    
    console.print(f"[green]✓[/green] Exported to {output}")


@main.command()
@click.option("--repo", "-r", default=".", help="Repository path")
def init(repo: str):
    """Initialize hooks in a repository."""
    from agentic.hooks.install import install_hooks
    
    repo_path = Path(repo).resolve()
    install_hooks(repo_path)
    console.print(f"[green]✓[/green] Hooks installed in {repo_path / '.github' / 'hooks'}")


if __name__ == "__main__":
    main()
