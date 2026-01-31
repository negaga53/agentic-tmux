"""Redis client abstraction for agentic-tmux with optional SQLite fallback."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Generator

from agentic.config import get_db_path, get_debug_log, ensure_config_dir, WORKING_DIR_ENV_VAR


def _get_debug_log_path(working_dir: str | None = None) -> Path:
    """Get the debug log path for the given working directory."""
    if working_dir is None:
        working_dir = os.environ.get(WORKING_DIR_ENV_VAR) or os.getcwd()
    return get_debug_log(working_dir, "sqlite_debug")


def _log_debug(msg: str, working_dir: str | None = None) -> None:
    """Write debug message to log file."""
    debug_log = _get_debug_log_path(working_dir)
    ensure_config_dir(working_dir)
    with open(debug_log, "a") as f:
        f.write(f"{time.time()}: {msg}\n")

import redis
import redis.asyncio as aioredis

from agentic.models import (
    ActionLog,
    Agent,
    AgentStatus,
    AgenticSession,
    ErrorReport,
    HeartbeatData,
    SessionStatus,
    Task,
    TaskDAG,
    TaskStatus,
)


class RedisKeys:
    """Redis key templates for agentic-tmux."""

    # Session-level keys
    @staticmethod
    def session_config(session_id: str) -> str:
        return f"session:{session_id}:config"

    @staticmethod
    def session_dag(session_id: str) -> str:
        return f"session:{session_id}:dag"

    @staticmethod
    def session_agents(session_id: str) -> str:
        return f"session:{session_id}:agents"

    @staticmethod
    def session_done(session_id: str) -> str:
        return f"session:{session_id}:done"

    # Per-agent keys
    @staticmethod
    def agent_config(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:config"

    @staticmethod
    def agent_queue(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:queue"

    @staticmethod
    def agent_status(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:status"

    @staticmethod
    def agent_log(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:log"

    @staticmethod
    def agent_heartbeat(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:heartbeat"

    @staticmethod
    def agent_result(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:result"

    # Communication keys
    @staticmethod
    def bus_broadcast(session_id: str) -> str:
        return f"bus:{session_id}:broadcast"

    @staticmethod
    def bus_admin(session_id: str) -> str:
        return f"bus:{session_id}:admin"

    # Inter-agent message queue
    @staticmethod
    def agent_messages(session_id: str, agent_id: str) -> str:
        return f"agent:{session_id}:{agent_id}:messages"

    # Pending plan keys (for MCP workflow)
    @staticmethod
    def pending_plan(plan_id: str) -> str:
        return f"plan:pending:{plan_id}"


class SQLiteAgenticClient:
    """SQLite client for agentic-tmux when Redis is not available.
    
    Uses SQLite for persistent storage that works across processes.
    All tmux panes can share state via the database file.
    """

    def __init__(self, db_path: Path | str | None = None, working_dir: str | None = None):
        """Initialize SQLite client.
        
        Args:
            db_path: Explicit database path (overrides working_dir)
            working_dir: Working directory for per-repo storage
        """
        if db_path:
            self._db_path = Path(db_path)
        else:
            self._db_path = get_db_path(working_dir)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise

    def _execute_with_retry(
        self, 
        operation: callable, 
        max_retries: int = 5
    ):
        """Execute a database operation with retry on busy."""
        for attempt in range(max_retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as cur:
            # Key-value store
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL
                )
            """)
            # Sets (for session agents)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sets (
                    key TEXT,
                    member TEXT,
                    PRIMARY KEY (key, member)
                )
            """)
            # Lists/Queues (for task queues)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS queues (
                    key TEXT,
                    idx INTEGER,
                    value TEXT,
                    PRIMARY KEY (key, idx)
                )
            """)
            # Streams (for logs)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS streams (
                    key TEXT,
                    entry_id TEXT,
                    data TEXT,
                    PRIMARY KEY (key, entry_id)
                )
            """)
            # Index for faster queue operations
            cur.execute("CREATE INDEX IF NOT EXISTS idx_queues_key ON queues(key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_streams_key ON streams(key)")

    def ping(self) -> bool:
        """Always returns True for SQLite storage."""
        return True

    def _cleanup_expired(self) -> None:
        """Remove expired keys."""
        now = time.time()
        with self._transaction() as cur:
            cur.execute("DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))

    # Session operations
    def create_session(self, session: AgenticSession) -> None:
        """Create a new session."""
        key = RedisKeys.session_config(session.id)
        data = self._serialize_hash(session.to_config())
        with self._transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, json.dumps(data)),
            )
            # Clear agents set
            agents_key = RedisKeys.session_agents(session.id)
            cur.execute("DELETE FROM sets WHERE key = ?", (agents_key,))

    def get_session(self, session_id: str) -> AgenticSession | None:
        """Get session by ID."""
        self._cleanup_expired()
        key = RedisKeys.session_config(session_id)
        
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            data = json.loads(row["value"])
            
            # Get agents
            agents_key = RedisKeys.session_agents(session_id)
            cur.execute("SELECT member FROM sets WHERE key = ?", (agents_key,))
            agent_ids = [r["member"] for r in cur.fetchall()]
        
        agents = {}
        for aid in agent_ids:
            agent = self.get_agent(session_id, aid)
            if agent:
                agents[aid] = agent

        dag = self.get_dag(session_id)

        # Parse config from JSON string if needed
        config = data.get("config", {})
        if isinstance(config, str):
            try:
                config = json.loads(config) if config else {}
            except json.JSONDecodeError:
                config = {}

        return AgenticSession(
            id=data.get("id", session_id),
            status=SessionStatus(data.get("status", "initializing")),
            admin_pane_id=data.get("admin_pane_id"),
            agents=agents,
            dag=dag or TaskDAG(),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            working_directory=data.get("working_directory", "."),
            config=config,
        )

    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status."""
        key = RedisKeys.session_config(session_id)
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                data = json.loads(row["value"])
                data["status"] = status.value
                data["updated_at"] = str(time.time())
                cur.execute("UPDATE kv SET value = ? WHERE key = ?", (json.dumps(data), key))

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        agents = self.get_all_agents(session_id)
        for agent in agents:
            self.delete_agent(session_id, agent.id)
        
        with self._transaction() as cur:
            # Delete session keys
            keys = [
                RedisKeys.session_config(session_id),
                RedisKeys.session_dag(session_id),
                RedisKeys.session_done(session_id),
            ]
            for key in keys:
                cur.execute("DELETE FROM kv WHERE key = ?", (key,))
            
            # Delete sets
            cur.execute("DELETE FROM sets WHERE key = ?", (RedisKeys.session_agents(session_id),))
            
            # Delete queues and streams matching session
            cur.execute("DELETE FROM queues WHERE key LIKE ?", (f"%{session_id}%",))
            cur.execute("DELETE FROM streams WHERE key LIKE ?", (f"%{session_id}%",))

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        with self._transaction() as cur:
            cur.execute("SELECT key FROM kv WHERE key LIKE 'session:%:config'")
            return [row["key"].split(":")[1] for row in cur.fetchall()]

    # Agent operations
    def register_agent(self, session_id: str, agent: Agent) -> None:
        """Register an agent in the session."""
        agents_key = RedisKeys.session_agents(session_id)
        config_key = RedisKeys.agent_config(session_id, agent.id)
        
        with self._transaction() as cur:
            # Add to agents set
            cur.execute(
                "INSERT OR REPLACE INTO sets (key, member) VALUES (?, ?)",
                (agents_key, agent.id),
            )
            # Store config
            cur.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (config_key, json.dumps(self._serialize_hash(agent.to_config()))),
            )
        
        self.set_agent_status(session_id, agent.id, AgentStatus.IDLE)
        self.update_heartbeat(session_id, agent.id)

    def get_agent(self, session_id: str, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        key = RedisKeys.agent_config(session_id, agent_id)
        
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            data = json.loads(row["value"])
        
        status = self.get_agent_status(session_id, agent_id)
        queue_len = self.get_queue_length(session_id, agent_id)
        heartbeat = self.get_heartbeat(session_id, agent_id)

        scope_str = data.get("scope", "{}")
        scope_data = json.loads(scope_str) if isinstance(scope_str, str) else scope_str
        
        return Agent(
            id=data.get("id", agent_id),
            role=data.get("role", "worker"),
            scope=scope_data,
            pane_id=data.get("pane_id"),
            status=status,
            model=data.get("model", "default"),
            created_at=float(data.get("created_at", time.time())),
            last_heartbeat=heartbeat,
            task_queue_length=queue_len,
        )

    def get_all_agents(self, session_id: str) -> list[Agent]:
        """Get all agents in a session."""
        agents_key = RedisKeys.session_agents(session_id)
        
        with self._transaction() as cur:
            cur.execute("SELECT member FROM sets WHERE key = ?", (agents_key,))
            agent_ids = [r["member"] for r in cur.fetchall()]
        
        agents = []
        for aid in agent_ids:
            agent = self.get_agent(session_id, aid)
            if agent:
                agents.append(agent)
        return agents

    def delete_agent(self, session_id: str, agent_id: str) -> None:
        """Delete an agent and its data."""
        agents_key = RedisKeys.session_agents(session_id)
        
        with self._transaction() as cur:
            # Remove from set
            cur.execute("DELETE FROM sets WHERE key = ? AND member = ?", (agents_key, agent_id))
            
            # Delete agent keys
            keys = [
                RedisKeys.agent_config(session_id, agent_id),
                RedisKeys.agent_status(session_id, agent_id),
                RedisKeys.agent_heartbeat(session_id, agent_id),
                RedisKeys.agent_result(session_id, agent_id),
            ]
            for key in keys:
                cur.execute("DELETE FROM kv WHERE key = ?", (key,))
            
            # Delete queue and logs
            cur.execute("DELETE FROM queues WHERE key = ?", (RedisKeys.agent_queue(session_id, agent_id),))
            cur.execute("DELETE FROM streams WHERE key = ?", (RedisKeys.agent_log(session_id, agent_id),))

    def set_agent_status(
        self, session_id: str, agent_id: str, status: AgentStatus, waiting_for: str | None = None
    ) -> None:
        """Set agent status."""
        key = RedisKeys.agent_status(session_id, agent_id)
        value = f"waiting:{waiting_for}" if status == AgentStatus.WAITING and waiting_for else status.value
        with self._transaction() as cur:
            cur.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))

    def get_agent_status(self, session_id: str, agent_id: str) -> AgentStatus:
        """Get agent status."""
        key = RedisKeys.agent_status(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return AgentStatus.IDLE
            status_str = row["value"]
            if status_str.startswith("waiting:"):
                return AgentStatus.WAITING
            return AgentStatus(status_str)

    def update_heartbeat(self, session_id: str, agent_id: str) -> None:
        """Update agent heartbeat timestamp."""
        key = RedisKeys.agent_heartbeat(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, str(time.time())))

    def get_heartbeat(self, session_id: str, agent_id: str) -> float:
        """Get agent heartbeat timestamp."""
        key = RedisKeys.agent_heartbeat(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            return float(row["value"]) if row else 0.0

    # Task queue operations
    def push_task(self, session_id: str, agent_id: str, task: Task) -> None:
        """Push a task to an agent's queue."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                with self._transaction() as cur:
                    # Use a random component to avoid collisions
                    import random
                    cur.execute("SELECT MAX(idx) as max_idx FROM queues WHERE key = ?", (key,))
                    row = cur.fetchone()
                    base_idx = (row["max_idx"] or -1) + 1
                    # Add random offset to reduce collision probability
                    next_idx = base_idx + attempt + random.randint(0, 100)
                    cur.execute(
                        "INSERT INTO queues (key, idx, value) VALUES (?, ?, ?)",
                        (key, next_idx, json.dumps(task.to_queue_message())),
                    )
                return  # Success
            except sqlite3.IntegrityError:
                if attempt == max_retries - 1:
                    raise  # Re-raise on last attempt
                # Small delay with exponential backoff
                time.sleep(0.05 * (2 ** attempt))
                continue

    def pop_task(self, session_id: str, agent_id: str, timeout: int = 5) -> Task | None:
        """Pop a task from an agent's queue."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        with self._transaction() as cur:
            # Get oldest entry
            cur.execute("SELECT idx, value FROM queues WHERE key = ? ORDER BY idx LIMIT 1", (key,))
            row = cur.fetchone()
            if not row:
                return None
            
            # Delete it
            cur.execute("DELETE FROM queues WHERE key = ? AND idx = ?", (key, row["idx"]))
            
            msg = json.loads(row["value"])
            if msg.get("task") == "done":
                return Task.done_task()
            
            return Task(
                id=msg.get("task_id", ""),
                title=msg.get("task", ""),
                description=msg.get("description", ""),
                from_agent=msg.get("from"),
                files=msg.get("files", []),
                metadata=msg.get("metadata", {}),
            )

    def get_queue_length(self, session_id: str, agent_id: str) -> int:
        """Get the length of an agent's task queue."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM queues WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["cnt"] if row else 0

    def push_done_to_all(self, session_id: str) -> None:
        """Push done signal to all agents."""
        agents = self.get_all_agents(session_id)
        done_task = Task.done_task()
        for agent in agents:
            self.push_task(session_id, agent.id, done_task)
        
        key = RedisKeys.session_done(session_id)
        with self._transaction() as cur:
            cur.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, "1"))

    def is_session_done(self, session_id: str) -> bool:
        """Check if session is marked as done."""
        key = RedisKeys.session_done(session_id)
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["value"] == "1" if row else False

    # Inter-agent message operations
    def send_agent_message(
        self, session_id: str, from_agent: str, to_agent: str, message: str
    ) -> str:
        """Send a message from one agent to another. Returns message ID."""
        import random
        _log_debug(f"send_agent_message: {from_agent} -> {to_agent}, msg_len={len(message)}")
        key = RedisKeys.agent_messages(session_id, to_agent)
        msg_id = f"{int(time.time() * 1000)}-{from_agent}"
        data = {
            "id": msg_id,
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": time.time(),
            "read": False,
        }
        max_retries = 10
        for attempt in range(max_retries):
            try:
                _log_debug(f"send_agent_message attempt {attempt + 1}")
                with self._transaction() as cur:
                    cur.execute("SELECT MAX(idx) as max_idx FROM queues WHERE key = ?", (key,))
                    row = cur.fetchone()
                    base_idx = (row["max_idx"] or -1) + 1
                    next_idx = base_idx + attempt + random.randint(0, 100)
                    cur.execute(
                        "INSERT INTO queues (key, idx, value) VALUES (?, ?, ?)",
                        (key, next_idx, json.dumps(data)),
                    )
                _log_debug(f"send_agent_message SUCCESS: {msg_id}")
                return msg_id
            except (sqlite3.IntegrityError, sqlite3.OperationalError) as e:
                _log_debug(f"send_agent_message ERROR attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff
                time.sleep(0.1 * (2 ** attempt))
                continue
        return msg_id

    def receive_agent_message(
        self, session_id: str, agent_id: str, timeout: int = 0
    ) -> dict | None:
        """Receive the next message for an agent. Non-blocking if timeout=0."""
        _log_debug(f"receive_agent_message: agent={agent_id}, timeout={timeout}")
        key = RedisKeys.agent_messages(session_id, agent_id)
        start_time = time.time()
        retry_count = 0
        max_retries = 5
        poll_count = 0
        
        while True:
            poll_count += 1
            try:
                with self._transaction() as cur:
                    cur.execute(
                        "SELECT idx, value FROM queues WHERE key = ? ORDER BY idx LIMIT 1",
                        (key,),
                    )
                    row = cur.fetchone()
                    if row:
                        # Remove the message from queue
                        cur.execute("DELETE FROM queues WHERE key = ? AND idx = ?", (key, row["idx"]))
                        msg = json.loads(row["value"])
                        _log_debug(f"receive_agent_message: GOT message from {msg.get('from')}")
                        return msg
                    retry_count = 0  # Reset on successful query
            except sqlite3.OperationalError as e:
                _log_debug(f"receive_agent_message: OperationalError {e}, retry={retry_count}")
                if "database is locked" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    time.sleep(0.1 * (2 ** retry_count))
                    continue
                raise
            
            # If no timeout or timeout expired, return None
            if timeout == 0 or (time.time() - start_time) >= timeout:
                _log_debug(f"receive_agent_message: TIMEOUT after {poll_count} polls")
                return None
            
            # Poll with small delay
            time.sleep(0.5)

    def peek_agent_messages(
        self, session_id: str, agent_id: str, count: int = 10
    ) -> list[dict]:
        """Peek at messages without removing them."""
        key = RedisKeys.agent_messages(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute(
                "SELECT value FROM queues WHERE key = ? ORDER BY idx LIMIT ?",
                (key, count),
            )
            return [json.loads(row["value"]) for row in cur.fetchall()]

    def get_message_count(self, session_id: str, agent_id: str) -> int:
        """Get number of pending messages for an agent."""
        key = RedisKeys.agent_messages(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM queues WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["cnt"] if row else 0

    # DAG operations
    def save_dag(self, session_id: str, dag: TaskDAG) -> None:
        """Save task DAG."""
        key = RedisKeys.session_dag(session_id)
        with self._transaction() as cur:
            cur.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(dag.to_dict())))

    def get_dag(self, session_id: str) -> TaskDAG | None:
        """Get task DAG."""
        key = RedisKeys.session_dag(session_id)
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            return TaskDAG.from_dict(json.loads(row["value"]))

    def update_task_status(
        self, session_id: str, task_id: str, status: TaskStatus, result: dict | None = None
    ) -> None:
        """Update a task's status in the DAG."""
        dag = self.get_dag(session_id)
        if dag and task_id in dag.tasks:
            dag.tasks[task_id].status = status
            if status == TaskStatus.IN_PROGRESS:
                dag.tasks[task_id].started_at = time.time()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
                dag.tasks[task_id].completed_at = time.time()
            if result:
                dag.tasks[task_id].result = result
            self.save_dag(session_id, dag)

    # Action log operations
    def log_action(self, session_id: str, agent_id: str, log: ActionLog) -> None:
        """Log an agent action."""
        key = RedisKeys.agent_log(session_id, agent_id)
        entry_id = f"{int(time.time() * 1000)}-{id(log)}"
        data = {
            "action": log.action,
            "details": json.dumps(log.details),
            "file": log.file or "",
            "tool": log.tool or "",
            "timestamp": str(log.timestamp),
        }
        with self._transaction() as cur:
            cur.execute(
                "INSERT INTO streams (key, entry_id, data) VALUES (?, ?, ?)",
                (key, entry_id, json.dumps(data)),
            )
            # Keep last 1000 entries
            cur.execute("""
                DELETE FROM streams WHERE key = ? AND entry_id NOT IN (
                    SELECT entry_id FROM streams WHERE key = ? ORDER BY entry_id DESC LIMIT 1000
                )
            """, (key, key))

    def get_agent_logs(
        self, session_id: str, agent_id: str, count: int = 50, last_id: str = "0"
    ) -> list[ActionLog]:
        """Get recent logs for an agent."""
        key = RedisKeys.agent_log(session_id, agent_id)
        with self._transaction() as cur:
            cur.execute(
                "SELECT entry_id, data FROM streams WHERE key = ? ORDER BY entry_id DESC LIMIT ?",
                (key, count),
            )
            rows = cur.fetchall()
        
        logs = []
        for row in reversed(rows):  # Reverse to get chronological order
            data = json.loads(row["data"])
            logs.append(
                ActionLog(
                    timestamp=float(data.get("timestamp", time.time())),
                    agent_id=agent_id,
                    action=data.get("action", ""),
                    details=json.loads(data.get("details", "{}")),
                    file=data.get("file") or None,
                    tool=data.get("tool") or None,
                )
            )
        return logs

    # Admin communication
    def send_to_admin(self, session_id: str, message: dict[str, Any]) -> None:
        """Send a message to the admin queue."""
        key = RedisKeys.bus_admin(session_id)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._transaction() as cur:
                    cur.execute("SELECT MAX(idx) as max_idx FROM queues WHERE key = ?", (key,))
                    row = cur.fetchone()
                    next_idx = (row["max_idx"] or -1) + 1
                    cur.execute(
                        "INSERT INTO queues (key, idx, value) VALUES (?, ?, ?)",
                        (key, next_idx, json.dumps(message)),
                    )
                return
            except sqlite3.IntegrityError:
                if attempt == max_retries - 1:
                    raise
                continue

    def receive_admin_message(self, session_id: str, timeout: int = 1) -> dict[str, Any] | None:
        """Receive a message from the admin queue."""
        key = RedisKeys.bus_admin(session_id)
        with self._transaction() as cur:
            cur.execute("SELECT idx, value FROM queues WHERE key = ? ORDER BY idx LIMIT 1", (key,))
            row = cur.fetchone()
            if not row:
                return None
            cur.execute("DELETE FROM queues WHERE key = ? AND idx = ?", (key, row["idx"]))
            return json.loads(row["value"])

    def report_error(self, session_id: str, error: ErrorReport) -> None:
        """Report an error to admin."""
        self.send_to_admin(
            session_id,
            {
                "type": "error",
                "agent_id": error.agent_id,
                "task_id": error.task_id,
                "error_type": error.error_type,
                "message": error.message,
                "retry_count": error.retry_count,
                "recoverable": error.recoverable,
                "timestamp": error.timestamp,
            },
        )

    def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message (stored for polling in SQLite mode)."""
        # Store in a broadcast queue that agents can poll
        key = f"broadcast:{session_id}"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._transaction() as cur:
                    cur.execute("SELECT MAX(idx) as max_idx FROM queues WHERE key = ?", (key,))
                    row = cur.fetchone()
                    next_idx = (row["max_idx"] or -1) + 1
                    cur.execute(
                        "INSERT INTO queues (key, idx, value) VALUES (?, ?, ?)",
                        (key, next_idx, json.dumps(message)),
                    )
                return
            except sqlite3.IntegrityError:
                if attempt == max_retries - 1:
                    raise
                continue

    # Pending plan operations
    def store_pending_plan(self, plan_id: str, plan: Any) -> None:
        """Store an execution plan temporarily for approval workflow."""
        from agentic.models import ExecutionPlan
        
        key = RedisKeys.pending_plan(plan_id)
        plan_data = {
            "prompt": plan.prompt,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "scope": {"patterns": a.scope.patterns, "read_only": a.scope.read_only},
                }
                for a in plan.agents
            ],
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "agent_id": t.agent_id,
                    "dependencies": t.dependencies,
                    "files": t.files,
                }
                for t in plan.dag.tasks.values()
            ],
            "communications": plan.estimated_communications,
        }
        expires_at = time.time() + 3600  # Expire after 1 hour
        with self._transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(plan_data), expires_at),
            )

    def get_pending_plan(self, plan_id: str) -> Any | None:
        """Retrieve a pending execution plan."""
        from agentic.models import Agent, ExecutionPlan, FileScope, Task, TaskDAG
        
        self._cleanup_expired()
        key = RedisKeys.pending_plan(plan_id)
        
        with self._transaction() as cur:
            cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            
            plan_data = json.loads(row["value"])
        
        agents = []
        for a in plan_data.get("agents", []):
            scope_data = a.get("scope", {})
            agents.append(Agent(
                id=a.get("id"),
                role=a.get("role"),
                scope=FileScope(
                    patterns=scope_data.get("patterns", ["**/*"]),
                    read_only=scope_data.get("read_only", False),
                ),
            ))
        
        dag = TaskDAG()
        for t in plan_data.get("tasks", []):
            task = Task(
                id=t.get("id", ""),
                title=t.get("title", ""),
                description=t.get("description", ""),
                agent_id=t.get("agent_id"),
                dependencies=t.get("dependencies", []),
                files=t.get("files", []),
            )
            dag.add_task(task)
        
        return ExecutionPlan(
            prompt=plan_data.get("prompt", ""),
            agents=agents,
            dag=dag,
            estimated_communications=plan_data.get("communications", []),
        )

    def delete_pending_plan(self, plan_id: str) -> None:
        """Delete a pending execution plan."""
        key = RedisKeys.pending_plan(plan_id)
        with self._transaction() as cur:
            cur.execute("DELETE FROM kv WHERE key = ?", (key,))

    def _serialize_hash(self, data: dict[str, Any]) -> dict[str, str]:
        """Serialize dict values for storage."""
        result = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                result[k] = json.dumps(v)
            elif v is None:
                result[k] = ""
            else:
                result[k] = str(v)
        return result


# Singleton for SQLite storage (keyed by working_dir)
# Using a dict to allow per-repo singletons
_sqlite_clients: dict[str, SQLiteAgenticClient] = {}
_sqlite_lock = threading.Lock()


def get_sqlite_client(
    db_path: Path | str | None = None,
    working_dir: str | None = None,
) -> SQLiteAgenticClient:
    """Get the singleton SQLite client instance for the given working directory.
    
    Args:
        db_path: Explicit database path (overrides working_dir-based path)
        working_dir: Working directory for per-repo storage. If None, uses CWD or env var.
    """
    global _sqlite_clients
    
    if working_dir is None:
        working_dir = os.environ.get(WORKING_DIR_ENV_VAR) or os.getcwd()
    
    # Normalize the working_dir path for consistency
    working_dir = os.path.abspath(working_dir)
    
    with _sqlite_lock:
        if working_dir not in _sqlite_clients:
            _sqlite_clients[working_dir] = SQLiteAgenticClient(db_path, working_dir)
        return _sqlite_clients[working_dir]


def reset_sqlite_client(working_dir: str | None = None) -> None:
    """Reset the SQLite client singleton for the given working directory.
    
    Call this after cleanup to ensure fresh connections to new database.
    
    Args:
        working_dir: Working directory. If None, uses CWD or env var.
    """
    global _sqlite_clients
    
    if working_dir is None:
        working_dir = os.environ.get(WORKING_DIR_ENV_VAR) or os.getcwd()
    
    working_dir = os.path.abspath(working_dir)
    
    with _sqlite_lock:
        if working_dir in _sqlite_clients:
            del _sqlite_clients[working_dir]


def get_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    prefer_redis: bool = True,
    working_dir: str | None = None,
) -> "AgenticRedisClient | SQLiteAgenticClient":
    """Get the appropriate storage client.
    
    If prefer_redis is True and Redis is available, returns a Redis client.
    Otherwise returns the SQLite client for persistent cross-process storage.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        prefer_redis: Whether to prefer Redis over SQLite
        working_dir: Working directory for per-repo SQLite storage
    """
    if prefer_redis:
        try:
            client = AgenticRedisClient(host=host, port=port, db=db)
            if client.ping():
                return client
        except Exception:
            pass
    
    return get_sqlite_client(working_dir=working_dir)


class AgenticRedisClient:
    """Synchronous Redis client for agentic-tmux."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._host = host
        self._port = port
        self._db = db

    def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    # Session operations
    def create_session(self, session: AgenticSession) -> None:
        """Create a new session in Redis."""
        key = RedisKeys.session_config(session.id)
        self.client.hset(key, mapping=self._serialize_hash(session.to_config()))
        # Initialize empty agent set
        self.client.delete(RedisKeys.session_agents(session.id))

    def get_session(self, session_id: str) -> AgenticSession | None:
        """Get session by ID."""
        key = RedisKeys.session_config(session_id)
        data = self.client.hgetall(key)
        if not data:
            return None
        
        # Get agents
        agent_ids = self.client.smembers(RedisKeys.session_agents(session_id))
        agents = {}
        for aid in agent_ids:
            agent = self.get_agent(session_id, aid)
            if agent:
                agents[aid] = agent

        # Get DAG
        dag = self.get_dag(session_id)

        # Parse config from JSON string if needed
        config = data.get("config", {})
        if isinstance(config, str):
            try:
                config = json.loads(config) if config else {}
            except json.JSONDecodeError:
                config = {}

        return AgenticSession(
            id=data.get("id", session_id),
            status=SessionStatus(data.get("status", "initializing")),
            admin_pane_id=data.get("admin_pane_id"),
            agents=agents,
            dag=dag or TaskDAG(),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            working_directory=data.get("working_directory", "."),
            config=config,
        )

    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status."""
        key = RedisKeys.session_config(session_id)
        self.client.hset(key, "status", status.value)
        self.client.hset(key, "updated_at", str(time.time()))

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        # Get all agent IDs
        agent_ids = self.client.smembers(RedisKeys.session_agents(session_id))
        
        # Delete agent keys
        for aid in agent_ids:
            self.delete_agent(session_id, aid)
        
        # Delete session keys
        keys_to_delete = [
            RedisKeys.session_config(session_id),
            RedisKeys.session_dag(session_id),
            RedisKeys.session_agents(session_id),
            RedisKeys.session_done(session_id),
            RedisKeys.bus_broadcast(session_id),
            RedisKeys.bus_admin(session_id),
        ]
        for key in keys_to_delete:
            self.client.delete(key)

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        keys = self.client.keys("session:*:config")
        return [k.split(":")[1] for k in keys]

    # Agent operations
    def register_agent(self, session_id: str, agent: Agent) -> None:
        """Register an agent in the session."""
        # Add to agent set
        self.client.sadd(RedisKeys.session_agents(session_id), agent.id)
        
        # Store agent config
        key = RedisKeys.agent_config(session_id, agent.id)
        self.client.hset(key, mapping=self._serialize_hash(agent.to_config()))
        
        # Initialize status
        self.set_agent_status(session_id, agent.id, AgentStatus.IDLE)
        
        # Initialize heartbeat
        self.update_heartbeat(session_id, agent.id)

    def get_agent(self, session_id: str, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        key = RedisKeys.agent_config(session_id, agent_id)
        data = self.client.hgetall(key)
        if not data:
            return None
        
        status = self.get_agent_status(session_id, agent_id)
        queue_len = self.client.llen(RedisKeys.agent_queue(session_id, agent_id))
        heartbeat = self.get_heartbeat(session_id, agent_id)

        # Parse scope
        scope_data = json.loads(data.get("scope", "{}"))
        
        return Agent(
            id=data.get("id", agent_id),
            role=data.get("role", "worker"),
            scope=scope_data,
            pane_id=data.get("pane_id"),
            status=status,
            model=data.get("model", "default"),
            created_at=float(data.get("created_at", time.time())),
            last_heartbeat=heartbeat,
            task_queue_length=queue_len,
        )

    def get_all_agents(self, session_id: str) -> list[Agent]:
        """Get all agents in a session."""
        agent_ids = self.client.smembers(RedisKeys.session_agents(session_id))
        agents = []
        for aid in agent_ids:
            agent = self.get_agent(session_id, aid)
            if agent:
                agents.append(agent)
        return agents

    def delete_agent(self, session_id: str, agent_id: str) -> None:
        """Delete an agent and its data."""
        # Remove from set
        self.client.srem(RedisKeys.session_agents(session_id), agent_id)
        
        # Delete agent keys
        keys_to_delete = [
            RedisKeys.agent_config(session_id, agent_id),
            RedisKeys.agent_queue(session_id, agent_id),
            RedisKeys.agent_status(session_id, agent_id),
            RedisKeys.agent_log(session_id, agent_id),
            RedisKeys.agent_heartbeat(session_id, agent_id),
            RedisKeys.agent_result(session_id, agent_id),
        ]
        for key in keys_to_delete:
            self.client.delete(key)

    def set_agent_status(
        self, session_id: str, agent_id: str, status: AgentStatus, waiting_for: str | None = None
    ) -> None:
        """Set agent status."""
        key = RedisKeys.agent_status(session_id, agent_id)
        if status == AgentStatus.WAITING and waiting_for:
            self.client.set(key, f"waiting:{waiting_for}")
        else:
            self.client.set(key, status.value)

    def get_agent_status(self, session_id: str, agent_id: str) -> AgentStatus:
        """Get agent status."""
        key = RedisKeys.agent_status(session_id, agent_id)
        status_str = self.client.get(key)
        if not status_str:
            return AgentStatus.IDLE
        if status_str.startswith("waiting:"):
            return AgentStatus.WAITING
        return AgentStatus(status_str)

    def update_heartbeat(self, session_id: str, agent_id: str) -> None:
        """Update agent heartbeat timestamp."""
        key = RedisKeys.agent_heartbeat(session_id, agent_id)
        self.client.set(key, str(time.time()))

    def get_heartbeat(self, session_id: str, agent_id: str) -> float:
        """Get agent heartbeat timestamp."""
        key = RedisKeys.agent_heartbeat(session_id, agent_id)
        ts = self.client.get(key)
        return float(ts) if ts else 0.0

    # Task queue operations
    def push_task(self, session_id: str, agent_id: str, task: Task) -> None:
        """Push a task to an agent's queue."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        self.client.lpush(key, json.dumps(task.to_queue_message()))

    def pop_task(self, session_id: str, agent_id: str, timeout: int = 5) -> Task | None:
        """Pop a task from an agent's queue (blocking)."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        result = self.client.brpop(key, timeout=timeout)
        if not result:
            return None
        _, data = result
        msg = json.loads(data)
        
        # Check for done signal
        if msg.get("task") == "done":
            return Task.done_task()
        
        return Task(
            id=msg.get("task_id", ""),
            title=msg.get("task", ""),
            description=msg.get("description", ""),
            from_agent=msg.get("from"),
            files=msg.get("files", []),
            metadata=msg.get("metadata", {}),
        )

    def get_queue_length(self, session_id: str, agent_id: str) -> int:
        """Get the length of an agent's task queue."""
        key = RedisKeys.agent_queue(session_id, agent_id)
        return self.client.llen(key)

    def push_done_to_all(self, session_id: str) -> None:
        """Push done signal to all agents."""
        agent_ids = self.client.smembers(RedisKeys.session_agents(session_id))
        done_task = Task.done_task()
        for aid in agent_ids:
            self.push_task(session_id, aid, done_task)
        # Set session done flag
        self.client.set(RedisKeys.session_done(session_id), "1")

    def is_session_done(self, session_id: str) -> bool:
        """Check if session is marked as done."""
        return self.client.get(RedisKeys.session_done(session_id)) == "1"

    # Inter-agent message operations
    def send_agent_message(
        self, session_id: str, from_agent: str, to_agent: str, message: str
    ) -> str:
        """Send a message from one agent to another. Returns message ID."""
        key = RedisKeys.agent_messages(session_id, to_agent)
        msg_id = f"{int(time.time() * 1000)}-{from_agent}"
        data = {
            "id": msg_id,
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": time.time(),
            "read": False,
        }
        self.client.lpush(key, json.dumps(data))
        return msg_id

    def receive_agent_message(
        self, session_id: str, agent_id: str, timeout: int = 0
    ) -> dict | None:
        """Receive the next message for an agent. Non-blocking if timeout=0."""
        key = RedisKeys.agent_messages(session_id, agent_id)
        if timeout == 0:
            result = self.client.rpop(key)
            if result:
                return json.loads(result)
            return None
        result = self.client.brpop(key, timeout=timeout)
        if not result:
            return None
        _, data = result
        return json.loads(data)

    def peek_agent_messages(
        self, session_id: str, agent_id: str, count: int = 10
    ) -> list[dict]:
        """Peek at messages without removing them."""
        key = RedisKeys.agent_messages(session_id, agent_id)
        items = self.client.lrange(key, -count, -1)
        return [json.loads(item) for item in reversed(items)]

    def get_message_count(self, session_id: str, agent_id: str) -> int:
        """Get number of pending messages for an agent."""
        key = RedisKeys.agent_messages(session_id, agent_id)
        return self.client.llen(key)

    # DAG operations
    def save_dag(self, session_id: str, dag: TaskDAG) -> None:
        """Save task DAG to Redis."""
        key = RedisKeys.session_dag(session_id)
        self.client.set(key, json.dumps(dag.to_dict()))

    def get_dag(self, session_id: str) -> TaskDAG | None:
        """Get task DAG from Redis."""
        key = RedisKeys.session_dag(session_id)
        data = self.client.get(key)
        if not data:
            return None
        return TaskDAG.from_dict(json.loads(data))

    def update_task_status(
        self, session_id: str, task_id: str, status: TaskStatus, result: dict | None = None
    ) -> None:
        """Update a task's status in the DAG."""
        dag = self.get_dag(session_id)
        if dag and task_id in dag.tasks:
            dag.tasks[task_id].status = status
            if status == TaskStatus.IN_PROGRESS:
                dag.tasks[task_id].started_at = time.time()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
                dag.tasks[task_id].completed_at = time.time()
            if result:
                dag.tasks[task_id].result = result
            self.save_dag(session_id, dag)

    # Action log operations
    def log_action(self, session_id: str, agent_id: str, log: ActionLog) -> None:
        """Log an agent action to the stream."""
        key = RedisKeys.agent_log(session_id, agent_id)
        self.client.xadd(
            key,
            {
                "action": log.action,
                "details": json.dumps(log.details),
                "file": log.file or "",
                "tool": log.tool or "",
                "timestamp": str(log.timestamp),
            },
            maxlen=1000,  # Keep last 1000 entries
        )

    def get_agent_logs(
        self, session_id: str, agent_id: str, count: int = 50, last_id: str = "0"
    ) -> list[ActionLog]:
        """Get recent logs for an agent."""
        key = RedisKeys.agent_log(session_id, agent_id)
        entries = self.client.xrange(key, min=last_id, count=count)
        logs = []
        for entry_id, data in entries:
            logs.append(
                ActionLog(
                    timestamp=float(data.get("timestamp", time.time())),
                    agent_id=agent_id,
                    action=data.get("action", ""),
                    details=json.loads(data.get("details", "{}")),
                    file=data.get("file") or None,
                    tool=data.get("tool") or None,
                )
            )
        return logs

    # Admin communication
    def send_to_admin(self, session_id: str, message: dict[str, Any]) -> None:
        """Send a message to the admin queue."""
        key = RedisKeys.bus_admin(session_id)
        self.client.lpush(key, json.dumps(message))

    def receive_admin_message(self, session_id: str, timeout: int = 1) -> dict[str, Any] | None:
        """Receive a message from the admin queue."""
        key = RedisKeys.bus_admin(session_id)
        result = self.client.brpop(key, timeout=timeout)
        if not result:
            return None
        _, data = result
        return json.loads(data)

    def report_error(self, session_id: str, error: ErrorReport) -> None:
        """Report an error to admin."""
        self.send_to_admin(
            session_id,
            {
                "type": "error",
                "agent_id": error.agent_id,
                "task_id": error.task_id,
                "error_type": error.error_type,
                "message": error.message,
                "retry_count": error.retry_count,
                "recoverable": error.recoverable,
                "timestamp": error.timestamp,
            },
        )

    # Broadcast operations
    def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all agents."""
        key = RedisKeys.bus_broadcast(session_id)
        self.client.publish(key, json.dumps(message))

    # Pending plan operations (for MCP workflow)
    def store_pending_plan(self, plan_id: str, plan: Any) -> None:
        """Store an execution plan temporarily for approval workflow."""
        from agentic.models import ExecutionPlan
        
        key = RedisKeys.pending_plan(plan_id)
        plan_data = {
            "prompt": plan.prompt,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "scope": {"patterns": a.scope.patterns, "read_only": a.scope.read_only},
                }
                for a in plan.agents
            ],
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "agent_id": t.agent_id,
                    "dependencies": t.dependencies,
                    "files": t.files,
                }
                for t in plan.dag.tasks.values()
            ],
            "communications": plan.estimated_communications,
        }
        # Expire after 1 hour
        self.client.setex(key, 3600, json.dumps(plan_data))

    def get_pending_plan(self, plan_id: str) -> Any | None:
        """Retrieve a pending execution plan."""
        from agentic.models import Agent, ExecutionPlan, FileScope, Task, TaskDAG
        
        key = RedisKeys.pending_plan(plan_id)
        data = self.client.get(key)
        if not data:
            return None
        
        plan_data = json.loads(data)
        
        # Reconstruct the plan
        agents = []
        for a in plan_data.get("agents", []):
            scope_data = a.get("scope", {})
            agents.append(Agent(
                id=a.get("id"),
                role=a.get("role"),
                scope=FileScope(
                    patterns=scope_data.get("patterns", ["**/*"]),
                    read_only=scope_data.get("read_only", False),
                ),
            ))
        
        dag = TaskDAG()
        for t in plan_data.get("tasks", []):
            task = Task(
                id=t.get("id", ""),
                title=t.get("title", ""),
                description=t.get("description", ""),
                agent_id=t.get("agent_id"),
                dependencies=t.get("dependencies", []),
                files=t.get("files", []),
            )
            dag.add_task(task)
        
        return ExecutionPlan(
            prompt=plan_data.get("prompt", ""),
            agents=agents,
            dag=dag,
            estimated_communications=plan_data.get("communications", []),
        )

    def delete_pending_plan(self, plan_id: str) -> None:
        """Delete a pending execution plan."""
        key = RedisKeys.pending_plan(plan_id)
        self.client.delete(key)

    # Utility
    def _serialize_hash(self, data: dict[str, Any]) -> dict[str, str]:
        """Serialize dict values for Redis hash storage."""
        result = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                result[k] = json.dumps(v)
            elif v is None:
                result[k] = ""
            else:
                result[k] = str(v)
        return result


class AsyncAgenticRedisClient:
    """Async Redis client for orchestrator daemon."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client: aioredis.Redis | None = None
        self._host = host
        self._port = port
        self._db = db

    async def connect(self) -> None:
        """Connect to Redis."""
        self.client = aioredis.Redis(
            host=self._host, port=self._port, db=self._db, decode_responses=True
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()

    async def ping(self) -> bool:
        """Check if Redis is available."""
        if not self.client:
            return False
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def get_all_agents(self, session_id: str) -> list[dict[str, Any]]:
        """Get all agents in a session."""
        if not self.client:
            return []
        
        agent_ids = await self.client.smembers(RedisKeys.session_agents(session_id))
        agents = []
        
        for aid in agent_ids:
            config = await self.client.hgetall(RedisKeys.agent_config(session_id, aid))
            status = await self.client.get(RedisKeys.agent_status(session_id, aid))
            heartbeat = await self.client.get(RedisKeys.agent_heartbeat(session_id, aid))
            queue_len = await self.client.llen(RedisKeys.agent_queue(session_id, aid))
            
            agents.append({
                "id": aid,
                "role": config.get("role", "worker"),
                "pane_id": config.get("pane_id"),
                "status": status or "idle",
                "heartbeat": float(heartbeat) if heartbeat else 0.0,
                "queue_length": queue_len,
            })
        
        return agents

    async def get_dag(self, session_id: str) -> TaskDAG | None:
        """Get task DAG from Redis."""
        if not self.client:
            return None
        
        key = RedisKeys.session_dag(session_id)
        data = await self.client.get(key)
        if not data:
            return None
        return TaskDAG.from_dict(json.loads(data))

    async def is_session_done(self, session_id: str) -> bool:
        """Check if session is marked as done."""
        if not self.client:
            return False
        return await self.client.get(RedisKeys.session_done(session_id)) == "1"

    async def push_done_to_all(self, session_id: str) -> None:
        """Push done signal to all agents."""
        if not self.client:
            return
        
        agent_ids = await self.client.smembers(RedisKeys.session_agents(session_id))
        done_msg = json.dumps(Task.done_task().to_queue_message())
        
        for aid in agent_ids:
            await self.client.lpush(RedisKeys.agent_queue(session_id, aid), done_msg)
        
        await self.client.set(RedisKeys.session_done(session_id), "1")

    async def all_queues_empty(self, session_id: str) -> bool:
        """Check if all agent queues are empty."""
        if not self.client:
            return True
        
        agent_ids = await self.client.smembers(RedisKeys.session_agents(session_id))
        
        for aid in agent_ids:
            queue_len = await self.client.llen(RedisKeys.agent_queue(session_id, aid))
            if queue_len > 0:
                return False
        
        return True

    async def send_to_admin(self, session_id: str, message: dict[str, Any]) -> None:
        """Send a message to the admin queue."""
        if not self.client:
            return
        key = RedisKeys.bus_admin(session_id)
        await self.client.lpush(key, json.dumps(message))

    async def subscribe_broadcast(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to broadcast messages."""
        if not self.client:
            return
        
        pubsub = self.client.pubsub()
        await pubsub.subscribe(RedisKeys.bus_broadcast(session_id))
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])
