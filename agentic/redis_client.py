"""Redis client abstraction for agentic-tmux."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

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

        return AgenticSession(
            id=data.get("id", session_id),
            status=SessionStatus(data.get("status", "initializing")),
            admin_pane_id=data.get("admin_pane_id"),
            agents=agents,
            dag=dag or TaskDAG(),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            working_directory=data.get("working_directory", "."),
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
