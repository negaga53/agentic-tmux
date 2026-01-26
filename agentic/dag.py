"""DAG operations and utilities for agentic-tmux."""

from __future__ import annotations

from typing import Any

from agentic.models import Agent, Task, TaskDAG, TaskStatus


def validate_dag(dag: TaskDAG) -> tuple[bool, list[str]]:
    """
    Validate a task DAG.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: list[str] = []
    
    # Check for cycles
    if dag.has_cycle():
        errors.append("DAG contains a cycle - circular dependencies detected")
    
    # Check for missing dependencies
    for task_id, task in dag.tasks.items():
        for dep_id in task.dependencies:
            if dep_id not in dag.tasks:
                errors.append(f"Task '{task_id}' depends on non-existent task '{dep_id}'")
    
    # Check for orphan tasks (no path from a root)
    roots = [tid for tid, t in dag.tasks.items() if not t.dependencies]
    if not roots and dag.tasks:
        errors.append("DAG has no root tasks (all tasks have dependencies)")
    
    # Check for unassigned tasks
    for task_id, task in dag.tasks.items():
        if task.agent_id is None:
            errors.append(f"Task '{task_id}' has no assigned agent")
    
    return len(errors) == 0, errors


def topological_sort(dag: TaskDAG) -> list[str]:
    """
    Return task IDs in topological order (dependencies first).
    
    Raises:
        ValueError: If the DAG contains a cycle.
    """
    if dag.has_cycle():
        raise ValueError("Cannot topologically sort a DAG with cycles")
    
    visited: set[str] = set()
    result: list[str] = []
    
    def dfs(task_id: str) -> None:
        if task_id in visited:
            return
        visited.add(task_id)
        
        task = dag.tasks.get(task_id)
        if task:
            for dep_id in task.dependencies:
                dfs(dep_id)
        
        result.append(task_id)
    
    for task_id in dag.tasks:
        dfs(task_id)
    
    return result


def get_critical_path(dag: TaskDAG) -> list[str]:
    """
    Find the critical path (longest dependency chain) in the DAG.
    
    Returns:
        List of task IDs in the critical path.
    """
    if dag.has_cycle():
        return []
    
    # Calculate longest path to each node
    longest_path: dict[str, list[str]] = {}
    
    sorted_tasks = topological_sort(dag)
    
    for task_id in sorted_tasks:
        task = dag.tasks.get(task_id)
        if not task:
            continue
        
        if not task.dependencies:
            longest_path[task_id] = [task_id]
        else:
            best_path: list[str] = []
            for dep_id in task.dependencies:
                dep_path = longest_path.get(dep_id, [])
                if len(dep_path) > len(best_path):
                    best_path = dep_path
            longest_path[task_id] = best_path + [task_id]
    
    # Find the longest path
    critical = []
    for path in longest_path.values():
        if len(path) > len(critical):
            critical = path
    
    return critical


def get_parallel_groups(dag: TaskDAG) -> list[list[str]]:
    """
    Group tasks that can be executed in parallel.
    
    Returns:
        List of groups, where each group contains task IDs that can run together.
    """
    if dag.has_cycle():
        return []
    
    groups: list[list[str]] = []
    remaining = set(dag.tasks.keys())
    completed: set[str] = set()
    
    while remaining:
        # Find tasks whose dependencies are all completed
        ready = []
        for task_id in remaining:
            task = dag.tasks[task_id]
            deps_done = all(dep_id in completed for dep_id in task.dependencies)
            if deps_done:
                ready.append(task_id)
        
        if not ready:
            # This shouldn't happen if there's no cycle
            break
        
        groups.append(ready)
        for task_id in ready:
            remaining.remove(task_id)
            completed.add(task_id)
    
    return groups


def detect_wait_cycle(
    agents: list[Agent],
    get_waiting_for: callable,
) -> list[str] | None:
    """
    Detect if there's a circular wait among agents.
    
    Args:
        agents: List of agents
        get_waiting_for: Function that returns the agent ID being waited on
    
    Returns:
        List of agent IDs in the cycle, or None if no cycle.
    """
    waiting_graph: dict[str, str | None] = {}
    
    for agent in agents:
        if agent.status.value.startswith("waiting"):
            waiting_graph[agent.id] = get_waiting_for(agent)
        else:
            waiting_graph[agent.id] = None
    
    # Detect cycle using DFS
    for start_id in waiting_graph:
        visited: set[str] = set()
        path: list[str] = []
        current = start_id
        
        while current and current not in visited:
            visited.add(current)
            path.append(current)
            current = waiting_graph.get(current)
        
        if current and current in path:
            # Found a cycle - extract it
            cycle_start = path.index(current)
            return path[cycle_start:] + [current]
    
    return None


def estimate_completion_time(
    dag: TaskDAG,
    avg_task_duration: float = 60.0,  # seconds
) -> float:
    """
    Estimate total completion time assuming parallel execution.
    
    Args:
        dag: The task DAG
        avg_task_duration: Average duration per task in seconds
    
    Returns:
        Estimated total time in seconds.
    """
    groups = get_parallel_groups(dag)
    # Each group executes in parallel, but groups are sequential
    return len(groups) * avg_task_duration


def assign_tasks_to_agents(
    dag: TaskDAG,
    agents: list[Agent],
) -> dict[str, list[str]]:
    """
    Assign tasks to agents based on their roles and file scopes.
    
    Returns:
        Dict mapping agent_id to list of task_ids.
    """
    assignments: dict[str, list[str]] = {agent.id: [] for agent in agents}
    
    for task_id, task in dag.tasks.items():
        if task.agent_id:
            # Already assigned
            if task.agent_id in assignments:
                assignments[task.agent_id].append(task_id)
            continue
        
        # Find best agent for this task
        best_agent = None
        best_score = -1
        
        for agent in agents:
            score = 0
            
            # Check if task files match agent scope
            for file in task.files:
                if agent.scope.matches(file):
                    score += 10
            
            # Check if task title/description mentions agent role
            if agent.role.lower() in task.title.lower():
                score += 5
            if agent.role.lower() in task.description.lower():
                score += 3
            
            # Prefer agents with fewer assigned tasks (load balancing)
            score -= len(assignments[agent.id])
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        if best_agent:
            task.agent_id = best_agent.id
            assignments[best_agent.id].append(task_id)
    
    return assignments


def create_task_flow_diagram(dag: TaskDAG) -> str:
    """
    Create an ASCII art diagram of the task flow.
    
    Returns:
        ASCII art string representation.
    """
    if not dag.tasks:
        return "  (empty DAG)"
    
    groups = get_parallel_groups(dag)
    lines = []
    
    for i, group in enumerate(groups):
        # Create boxes for this level
        row_boxes = []
        for task_id in group:
            task = dag.tasks[task_id]
            agent_str = f"({task.agent_id})" if task.agent_id else ""
            label = f"{task_id}: {task.title[:15]}{agent_str}"
            box = f"┌{'─' * (len(label) + 2)}┐\n│ {label} │\n└{'─' * (len(label) + 2)}┘"
            row_boxes.append(box)
        
        # Join boxes horizontally
        if row_boxes:
            box_lines = [box.split("\n") for box in row_boxes]
            max_height = max(len(bl) for bl in box_lines)
            
            for line_idx in range(max_height):
                row_str = "    ".join(
                    bl[line_idx] if line_idx < len(bl) else " " * len(bl[0])
                    for bl in box_lines
                )
                lines.append("    " + row_str)
        
        # Add arrows to next level
        if i < len(groups) - 1:
            lines.append("        │")
            lines.append("        ▼")
    
    return "\n".join(lines)


def merge_dags(dag1: TaskDAG, dag2: TaskDAG, connect_to: str | None = None) -> TaskDAG:
    """
    Merge two DAGs, optionally connecting dag2 to a specific task in dag1.
    
    Args:
        dag1: First DAG
        dag2: Second DAG to merge in
        connect_to: Task ID in dag1 that dag2 roots should depend on
    
    Returns:
        Merged DAG.
    """
    merged = TaskDAG()
    
    # Add all tasks from dag1
    for tid, task in dag1.tasks.items():
        merged.tasks[tid] = task.model_copy()
    
    # Add all tasks from dag2
    for tid, task in dag2.tasks.items():
        new_task = task.model_copy()
        
        # If connecting to a specific task, add dependency to root tasks
        if connect_to and not task.dependencies:
            new_task.dependencies = [connect_to]
        
        # Handle ID conflicts by prefixing
        new_id = tid
        if tid in merged.tasks:
            new_id = f"m_{tid}"
            new_task.id = new_id
        
        merged.tasks[new_id] = new_task
    
    return merged
