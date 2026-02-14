"""Pydantic models for agentic-tmux."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    POLLING = "polling"  # Waiting for messages in receive_message loop
    FAILED = "failed"
    DONE = "done"


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SessionStatus(str, Enum):
    """Session status enumeration."""

    INITIALIZING = "initializing"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class FileScope(BaseModel):
    """Defines file access permissions for an agent."""

    patterns: list[str] = Field(default_factory=list, description="Glob patterns for allowed paths")
    read_only: bool = Field(default=False, description="If true, agent can only read files")

    def matches(self, file_path: str) -> bool:
        """Check if a file path matches any of the scope patterns."""
        import fnmatch

        for pattern in self.patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False


class Task(BaseModel):
    """Represents a task to be executed by an agent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str = ""
    agent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    from_agent: str | None = Field(default=None, description="Agent that created this task")
    files: list[str] = Field(default_factory=list, description="Files involved in this task")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_queue_message(self) -> dict[str, Any]:
        """Convert task to a message for Redis queue."""
        return {
            "task_id": self.id,
            "task": self.title,
            "description": self.description,
            "from": self.from_agent,
            "files": self.files,
            "metadata": self.metadata,
        }

    @classmethod
    def done_task(cls) -> Task:
        """Create a special 'done' task to signal agent shutdown."""
        return cls(id="done", title="done", description="Shutdown signal")


class Agent(BaseModel):
    """Represents an agent running in a tmux pane."""

    id: str = Field(default_factory=lambda: f"W{uuid.uuid4().hex[:4]}")
    role: str
    scope: FileScope = Field(default_factory=FileScope)
    pane_id: str | None = None
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: str | None = None
    waiting_for: str | None = Field(default=None, description="Agent ID if waiting for another agent")
    created_at: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)
    task_queue_length: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    model: str = "default"

    def to_config(self) -> dict[str, Any]:
        """Convert agent to config dict for Redis."""
        return {
            "id": self.id,
            "role": self.role,
            "scope": self.scope.model_dump(),
            "pane_id": self.pane_id,
            "model": self.model,
            "created_at": self.created_at,
        }


class TaskDAG(BaseModel):
    """Directed Acyclic Graph of tasks with dependencies."""

    tasks: dict[str, Task] = Field(default_factory=dict)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the DAG."""
        self.tasks[task.id] = task

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                dep_id in self.tasks
                and self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)
        return ready

    def is_complete(self) -> bool:
        """Check if all tasks are completed or skipped."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for task in self.tasks.values()
        )

    def has_cycle(self) -> bool:
        """Detect if there's a cycle in the dependency graph."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = self.tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True
        return False

    def get_completion_progress(self) -> tuple[int, int]:
        """Return (completed_count, total_count)."""
        completed = sum(
            1 for t in self.tasks.values()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        )
        return completed, len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize DAG to dict for storage."""
        return {
            "tasks": {tid: t.model_dump() for tid, t in self.tasks.items()}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskDAG:
        """Deserialize DAG from dict."""
        dag = cls()
        for tid, tdata in data.get("tasks", {}).items():
            dag.tasks[tid] = Task(**tdata)
        return dag


class ExecutionPlan(BaseModel):
    """Complete execution plan for a user prompt."""

    prompt: str
    agents: list[Agent]
    dag: TaskDAG
    estimated_communications: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class ActionLog(BaseModel):
    """Log entry for an agent action."""

    timestamp: float = Field(default_factory=time.time)
    agent_id: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    file: str | None = None
    tool: str | None = None


class AgenticSession(BaseModel):
    """Represents a full agentic session."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: SessionStatus = SessionStatus.INITIALIZING
    admin_pane_id: str | None = None
    agents: dict[str, Agent] = Field(default_factory=dict)
    dag: TaskDAG = Field(default_factory=TaskDAG)
    current_prompt: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    working_directory: str = "."
    config: dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        """Convert session to config dict for Redis."""
        return {
            "id": self.id,
            "status": self.status.value,
            "admin_pane_id": self.admin_pane_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "working_directory": self.working_directory,
            "config": self.config,
        }


class HeartbeatData(BaseModel):
    """Heartbeat data sent by agents."""

    agent_id: str
    timestamp: float = Field(default_factory=time.time)
    status: AgentStatus
    current_task_id: str | None = None
    queue_length: int = 0


class ErrorReport(BaseModel):
    """Error report for escalation to admin."""

    agent_id: str
    task_id: str | None = None
    error_type: str
    message: str
    retry_count: int = 0
    timestamp: float = Field(default_factory=time.time)
    recoverable: bool = True
