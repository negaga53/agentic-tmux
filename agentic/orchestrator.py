"""Orchestrator daemon for agentic-tmux."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic.dag import detect_wait_cycle
from agentic.models import Agent, AgentStatus, SessionStatus, TaskStatus
from agentic.redis_client import AsyncAgenticRedisClient, RedisKeys
from agentic.tmux_manager import TmuxManager

logger = logging.getLogger("agentic.orchestrator")


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator daemon."""

    session_id: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Monitoring intervals
    heartbeat_check_interval: float = 30.0  # seconds
    deadlock_check_interval: float = 15.0  # seconds
    task_check_interval: float = 5.0  # seconds
    
    # Thresholds
    heartbeat_timeout: float = 120.0  # seconds
    task_stuck_timeout: float = 600.0  # 10 minutes
    max_respawn_attempts: int = 3
    
    # Retry settings
    retry_base_delay: float = 5.0
    retry_max_delay: float = 45.0
    retry_multiplier: float = 3.0


@dataclass
class AgentState:
    """Runtime state for an agent."""

    agent_id: str
    pane_id: str | None = None
    status: str = "idle"
    current_task_id: str | None = None
    task_started_at: float | None = None
    last_heartbeat: float = field(default_factory=time.time)
    respawn_count: int = 0
    consecutive_failures: int = 0


class Orchestrator:
    """
    Background daemon that monitors and coordinates agents.
    
    Responsibilities:
    - Monitor agent heartbeats and respawn dead panes
    - Detect and break deadlocks
    - Track task completion and signal when done
    - Escalate failures to admin
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.redis = AsyncAgenticRedisClient(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
        )
        self.tmux = TmuxManager()
        self.agent_states: dict[str, AgentState] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the orchestrator daemon."""
        logger.info(f"Starting orchestrator for session {self.config.session_id}")
        
        await self.redis.connect()
        
        if not await self.redis.ping():
            logger.error("Failed to connect to Redis")
            return
        
        self._running = True
        self._setup_signal_handlers()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._deadlock_monitor()),
            asyncio.create_task(self._task_completion_monitor()),
            asyncio.create_task(self._admin_message_handler()),
        ]
        
        logger.info("Orchestrator started")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.redis.close()
        
        logger.info("Orchestrator stopped")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._running = False
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def stop(self) -> None:
        """Stop the orchestrator daemon."""
        self._running = False
        self._shutdown_event.set()

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and respawn dead panes."""
        while self._running:
            try:
                await self._check_heartbeats()
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
            
            await asyncio.sleep(self.config.heartbeat_check_interval)

    async def _check_heartbeats(self) -> None:
        """Check all agent heartbeats."""
        agents = await self.redis.get_all_agents(self.config.session_id)
        current_time = time.time()
        
        for agent_data in agents:
            agent_id = agent_data["id"]
            heartbeat = agent_data.get("heartbeat", 0)
            pane_id = agent_data.get("pane_id")
            
            # Update local state
            if agent_id not in self.agent_states:
                self.agent_states[agent_id] = AgentState(
                    agent_id=agent_id,
                    pane_id=pane_id,
                )
            
            state = self.agent_states[agent_id]
            state.last_heartbeat = heartbeat
            state.status = agent_data.get("status", "idle")
            
            # Check if heartbeat is stale
            if current_time - heartbeat > self.config.heartbeat_timeout:
                logger.warning(f"Agent {agent_id} heartbeat timeout, attempting respawn")
                await self._respawn_agent(agent_id, pane_id)

    async def _respawn_agent(self, agent_id: str, pane_id: str | None) -> None:
        """Respawn a dead agent pane."""
        if agent_id not in self.agent_states:
            return
        
        state = self.agent_states[agent_id]
        
        if state.respawn_count >= self.config.max_respawn_attempts:
            logger.error(f"Agent {agent_id} exceeded max respawn attempts")
            await self._escalate_to_admin(
                "agent_dead",
                f"Agent {agent_id} failed to respawn after {state.respawn_count} attempts",
                agent_id=agent_id,
            )
            return
        
        if pane_id:
            success = self.tmux.respawn_pane(pane_id)
            if success:
                state.respawn_count += 1
                logger.info(f"Respawned agent {agent_id} (attempt {state.respawn_count})")
            else:
                logger.error(f"Failed to respawn agent {agent_id}")

    async def _deadlock_monitor(self) -> None:
        """Monitor for deadlocks among agents."""
        while self._running:
            try:
                await self._check_for_deadlocks()
            except Exception as e:
                logger.error(f"Error in deadlock monitor: {e}")
            
            await asyncio.sleep(self.config.deadlock_check_interval)

    async def _check_for_deadlocks(self) -> None:
        """Check for circular waits and stuck agents."""
        agents_data = await self.redis.get_all_agents(self.config.session_id)
        
        # Build agent list for cycle detection
        agents = []
        for data in agents_data:
            agent = Agent(
                id=data["id"],
                role=data.get("role", "worker"),
                status=self._parse_status(data.get("status", "idle")),
            )
            agents.append(agent)
        
        def get_waiting_for(agent: Agent) -> str | None:
            status = agent.status.value
            if status.startswith("waiting:"):
                return status.split(":")[1]
            return None
        
        # Check for circular waits
        cycle = detect_wait_cycle(agents, get_waiting_for)
        if cycle:
            logger.warning(f"Deadlock detected: {' → '.join(cycle)}")
            await self._break_deadlock(cycle)

    async def _break_deadlock(self, cycle: list[str]) -> None:
        """Break a deadlock by picking one agent to proceed."""
        if not cycle:
            return
        
        # Pick the first agent in the cycle to break the wait
        agent_to_unblock = cycle[0]
        
        logger.info(f"Breaking deadlock by unblocking agent {agent_to_unblock}")
        
        # Notify admin
        await self._escalate_to_admin(
            "deadlock_broken",
            f"Broke deadlock cycle: {' → '.join(cycle)}. Unblocked {agent_to_unblock}",
            agents=cycle,
        )

    async def _task_completion_monitor(self) -> None:
        """Monitor task completion and signal when done."""
        while self._running:
            try:
                await self._check_task_completion()
            except Exception as e:
                logger.error(f"Error in task completion monitor: {e}")
            
            await asyncio.sleep(self.config.task_check_interval)

    async def _check_task_completion(self) -> None:
        """Check if all tasks are complete."""
        # Check if session is already done
        if await self.redis.is_session_done(self.config.session_id):
            logger.info("Session already marked as done")
            await self.stop()
            return
        
        # Get DAG and check completion
        dag = await self.redis.get_dag(self.config.session_id)
        if not dag:
            return
        
        if dag.is_complete():
            # Also check that all queues are empty
            if await self.redis.all_queues_empty(self.config.session_id):
                logger.info("All tasks complete and queues empty, sending done signal")
                await self.redis.push_done_to_all(self.config.session_id)
                await self._escalate_to_admin(
                    "session_complete",
                    "All tasks have been completed successfully",
                )
                await self.stop()

    async def _admin_message_handler(self) -> None:
        """Handle messages from admin queue (for orchestrator commands)."""
        while self._running:
            try:
                # This is a placeholder - in a full implementation,
                # we'd have a way to receive commands from admin
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in admin message handler: {e}")

    async def _escalate_to_admin(
        self,
        event_type: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Send an escalation message to admin."""
        await self.redis.send_to_admin(
            self.config.session_id,
            {
                "type": event_type,
                "message": message,
                "timestamp": time.time(),
                **kwargs,
            },
        )

    def _parse_status(self, status_str: str) -> AgentStatus:
        """Parse status string to AgentStatus enum."""
        if status_str.startswith("waiting:"):
            return AgentStatus.WAITING
        try:
            return AgentStatus(status_str)
        except ValueError:
            return AgentStatus.IDLE


def run_orchestrator(
    session_id: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    log_level: str = "INFO",
    pid_file: str | None = None,
) -> None:
    """
    Run the orchestrator daemon.
    
    Args:
        session_id: The agentic session ID
        redis_host: Redis host
        redis_port: Redis port
        log_level: Logging level
        pid_file: Path to write PID file
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Write PID file
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))
    
    # Create and run orchestrator
    config = OrchestratorConfig(
        session_id=session_id,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    
    orchestrator = Orchestrator(config)
    
    try:
        asyncio.run(orchestrator.start())
    finally:
        # Cleanup PID file
        if pid_file and Path(pid_file).exists():
            Path(pid_file).unlink()


def start_orchestrator_background(
    session_id: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    log_file: str | None = None,
) -> int | None:
    """
    Start the orchestrator as a background process.
    
    Returns:
        PID of the background process, or None if failed.
    """
    import subprocess
    
    cmd = [
        sys.executable,
        "-m", "agentic.orchestrator",
        "--session-id", session_id,
        "--redis-host", redis_host,
        "--redis-port", str(redis_port),
    ]
    
    if log_file:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    
    return process.pid


def stop_orchestrator(pid_file: str) -> bool:
    """
    Stop a running orchestrator by PID file.
    
    Returns:
        True if stopped successfully.
    """
    try:
        pid = int(Path(pid_file).read_text().strip())
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to exit
        for _ in range(10):
            try:
                os.kill(pid, 0)  # Check if still running
                time.sleep(0.5)
            except ProcessLookupError:
                return True
        
        # Force kill if still running
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic orchestrator daemon")
    parser.add_argument("--session-id", required=True, help="Session ID")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--pid-file", help="PID file path")
    
    args = parser.parse_args()
    
    run_orchestrator(
        session_id=args.session_id,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        log_level=args.log_level,
        pid_file=args.pid_file,
    )
